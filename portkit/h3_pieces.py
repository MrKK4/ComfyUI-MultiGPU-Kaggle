"""CPU export of H3 TP weight pieces. Does not change the inference runtime."""
import ast
import contextlib
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile
import time
import types

import safetensors
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

from tp_native_layout import NativeLayoutFile, needs_wrapper

LAYOUT_VERSION = 1
PROJECTIONS = ('attn.qkv_proj', 'attn.out_proj', 'mlp.fc1', 'mlp.fc2')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def equal_bytes(actual, expected):
    return (actual.shape == expected.shape and actual.dtype == expected.dtype
            and torch.equal(actual.contiguous().reshape(-1).view(torch.uint8),
                            expected.contiguous().reshape(-1).view(torch.uint8)))


def open_source(path):
    handle = safe_open(str(path), framework='pt', device='cpu')
    return NativeLayoutFile(handle) if needs_wrapper(handle) else handle


class BlockWindow:
    def __init__(self, source, start, count):
        self.source, self.start, self.count = source, start, count

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def name(self, key):
        match = re.fullmatch(r'blocks\.(\d+)\.(.+)', key)
        if not match or not 0 <= int(match[1]) < self.count:
            raise ValueError('Invalid window key: ' + key)
        return f'blocks.{self.start + int(match[1])}.{match[2]}'

    def get_tensor(self, key):
        return self.source.get_tensor(self.name(key))

    def get_slice(self, key):
        return self.source.get_slice(self.name(key))

    def metadata(self):
        layers = json.loads(self.source.metadata()['_quantization_metadata'])['layers']
        remapped = {f'blocks.0.{name}': layers[f'blocks.{self.start}.{name}'] for name in PROJECTIONS}
        return {'_quantization_metadata': json.dumps({'layers': remapped})}


def reference_builder(reference):
    """Execute the actual splitter with CPU devices and its CUDA context removed."""
    text = Path(reference).read_text(encoding='utf-8')
    tree = ast.parse(text)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('_rows', '_load_shards')]
    if len(functions) != 2:
        raise ValueError('Reference must contain _rows and _load_shards')
    reference_hash = hashlib.sha256('\n'.join(ast.get_source_segment(text, f) for f in functions).encode()).hexdigest()

    class CPUContext(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if ast.unparse(node.func) == 'torch.cuda.device':
                return ast.copy_location(ast.Call(func=ast.Attribute(value=ast.Name(id='contextlib', ctx=ast.Load()),
                      attr='nullcontext', ctx=ast.Load()), args=[], keywords=[]), node)
            return node

    module = ast.fix_missing_locations(CPUContext().visit(ast.Module(body=functions, type_ignores=[])))
    namespace = dict(torch=torch, json=json, time=time, contextlib=contextlib, HEAD=128,
                     DEVICES=(torch.device('cpu'), torch.device('cpu')), _SHARD_CACHE={},
                     _complete=lambda cached, count: cached is not None and len(cached) == 2 and all(
                         rank is not None and len(rank) == count and all(block is not None for block in rank)
                         for rank in cached),
                     H3BundleSource=type('H3BundleSource', (), {}),
                     open_weights=lambda window: window, logger=logging.getLogger('h3-export'),
                     comfy=types.SimpleNamespace(model_management=types.SimpleNamespace(free_memory=lambda *args: None)))
    exec(compile(module, str(reference), 'exec'), namespace)
    return namespace, reference_hash


def validate_source(source):
    indices = sorted({int(match[1]) for key in source.keys()
                      if (match := re.match(r'^blocks\.(\d+)\.', key))})
    if not indices or indices != list(range(len(indices))):
        raise ValueError('Block indices must be contiguous from zero')
    layers = json.loads(source.metadata()['_quantization_metadata'])['layers']
    baseline = None
    for index in indices:
        prefix = f'blocks.{index}.'
        schema = {}
        for name in PROJECTIONS:
            config = layers[prefix + name]
            if (config.get('format') != 'int8_tensorwise' or config.get('convrot') is not True
                    or config.get('convrot_groupsize') != 256 or config.get('transposed', False)):
                raise ValueError(f'Unsupported quantization in {prefix + name}: {config}')
            weight = source.get_tensor(prefix + name + '.weight')
            scale = source.get_tensor(prefix + name + '.weight_scale')
            if weight.dtype != torch.int8 or weight.ndim != 2:
                raise ValueError('Expected int8 matrix: ' + prefix + name)
            if tuple(scale.shape) not in ((weight.shape[0],), (weight.shape[0], 1)) or not scale.is_floating_point():
                raise ValueError('Expected per-row floating scale: ' + prefix + name)
            schema[name] = (tuple(weight.shape), str(scale.dtype), tuple(scale.shape))
        hidden = schema['attn.qkv_proj'][0][1]
        inner = schema['attn.qkv_proj'][0][0] // 3
        ffn = schema['mlp.fc1'][0][0] // 2
        if (schema['attn.qkv_proj'][0][0] != 3 * inner or schema['mlp.fc1'][0][0] != 2 * ffn
                or schema['attn.out_proj'][0] != (hidden, inner)
                or schema['mlp.fc1'][0][1] != hidden or schema['mlp.fc2'][0] != (hidden, ffn)
                or hidden % 256 or inner % 512 or ffn % 512):
            raise ValueError('Unsupported TP dimensions in block ' + str(index))
        for name in ('norm1.weight', 'norm2.weight', 'attn.q_norm.weight', 'attn.k_norm.weight'):
            tensor = source.get_tensor(prefix + name)
            if tuple(tensor.shape) != ((hidden,) if name.startswith('norm') else (128,)):
                raise ValueError('Unsupported norm shape: ' + prefix + name)
            schema[name] = (tuple(tensor.shape), str(tensor.dtype))
        if baseline is not None and schema != baseline:
            raise ValueError('Nonuniform block schema: ' + str(index))
        baseline = schema
    return len(indices)


def verify_reconstruction(source, block_index, ranks):
    """Independent inverse checks; no use of _rows or the exported file schema."""
    prefix = f'blocks.{block_index}.'
    left, right = ranks
    for key, original, groups in (('qkv', 'attn.qkv_proj', 3), ('fc1', 'mlp.fc1', 2)):
        for element, suffix in ((0, '.weight'), (1, '.weight_scale')):
            a, b = left[key][element], right[key][element]
            width = a.shape[0] // groups
            pieces = [part for group in range(groups) for part in
                      (a[group*width:(group+1)*width], b[group*width:(group+1)*width])]
            if not equal_bytes(torch.cat(pieces), source.get_tensor(prefix + original + suffix)):
                raise ValueError(f'Reconstruction failed: block {block_index} {key} {element}')
    for key, original in (('out', 'attn.out_proj'), ('fc2', 'mlp.fc2')):
        if not equal_bytes(torch.cat((left[key][0], right[key][0]), dim=1), source.get_tensor(prefix + original + '.weight')):
            raise ValueError('Column reconstruction failed: ' + prefix + original)
        for rank in ranks:
            if not equal_bytes(rank[key][1], source.get_tensor(prefix + original + '.weight_scale')):
                raise ValueError('Replicated scale differs: ' + prefix + original)
    for rank in ranks:
        for key, original in (('norm1', 'norm1.weight'), ('norm2', 'norm2.weight'),
                              ('qn', 'attn.q_norm.weight'), ('kn', 'attn.k_norm.weight')):
            expected = source.get_tensor(prefix + original)
            if key.startswith('norm'):
                expected = expected.float()
            if not equal_bytes(rank[key], expected):
                raise ValueError('Replicated small tensor differs: ' + prefix + key)
        inner = source.get_slice(prefix + 'attn.qkv_proj.weight').get_shape()[0] // 6
        if rank['inner'] != inner or rank['heads'] != inner // 128:
            raise ValueError('Invalid heads/inner scalar metadata')


def flatten(blocks, start):
    tensors, scalars = {}, {}
    for index, block in enumerate(blocks, start):
        for name, value in block.items():
            key = f'blocks.{index}.{name}'
            if isinstance(value, tuple):
                for suffix, tensor in zip(('weight', 'scale'), value):
                    tensors[key + '.' + suffix] = tensor.detach().clone().contiguous()
            elif torch.is_tensor(value):
                tensors[key] = value.detach().clone().contiguous()
            else:
                scalars[key] = value
    return tensors, scalars


def tensor_schema(tensors):
    return {name: dict(shape=list(t.shape), dtype=str(t.dtype), nbytes=t.numel()*t.element_size())
            for name, t in tensors.items()}


def load_piece(directory, descriptor):
    filename = descriptor['file']
    if not re.fullmatch(r'rank[01]_blocks_\d+-\d+\.safetensors', filename):
        raise ValueError('Unsafe piece filename')
    root = Path(directory).resolve()
    path = root/filename
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError('Piece path escapes export directory')
    if path.stat().st_size != descriptor['file_bytes'] or sha256(path) != descriptor['sha256']:
        raise ValueError('Piece size or SHA-256 mismatch: ' + filename)
    with safe_open(str(path), framework='pt', device='cpu') as handle:
        metadata = handle.metadata()
        if (metadata.get('layout_version') != str(LAYOUT_VERSION)
                or metadata.get('rank') != str(descriptor['rank'])
                or json.loads(metadata.get('scalars', '{}')) != descriptor['scalars']):
            raise ValueError('Piece scalar/layout metadata mismatch: ' + filename)
    tensors = load_file(str(path), device='cpu')
    if tensor_schema(tensors) != descriptor['tensors']:
        raise ValueError('Piece tensor schema mismatch: ' + filename)
    return tensors


def export_pieces(source_path, output_root, reference, source_identity=None, blocks_per_piece=2, expected_blocks=50):
    source_path = Path(source_path).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    before = source_path.stat()
    digest = sha256(source_path)
    identity = dict(source_identity or {})
    if identity.get('sha256') and identity['sha256'] != digest:
        raise ValueError('Source file SHA-256 differs from pinned identity')
    if identity.get('size') is not None and identity['size'] != before.st_size:
        raise ValueError('Source file size differs from pinned identity')
    identity.update(sha256=digest, size=before.st_size)
    namespace, reference_hash = reference_builder(reference)
    exporter_hash = sha256(__file__)
    destination = output_root/f'{digest[:16]}-{exporter_hash[:12]}'
    if destination.exists():
        raise FileExistsError(destination)
    if shutil.disk_usage(output_root).free < int(before.st_size * 1.15):
        raise OSError('Insufficient free disk space for exported pieces')
    if not isinstance(blocks_per_piece, int) or blocks_per_piece < 1:
        raise ValueError('blocks_per_piece must be positive')
    temporary = Path(tempfile.mkdtemp(prefix='.h3-export-', dir=output_root))
    manifest = dict(layout_version=LAYOUT_VERSION, exporter_sha256=exporter_hash,
                    reference_sha256=reference_hash, source=identity, tp_degree=2,
                    convrot_group_size=256, head_dim=128, blocks_per_piece=blocks_per_piece,
                    torch_version=torch.__version__, safetensors_version=safetensors.__version__,
                    requires_source_for_nonblock_layers=True, gpu_consumer_validated=False, pieces=[])
    try:
        with open(source_path, 'rb') as handle:
            length_bytes = handle.read(8)
            header_bytes = handle.read(struct.unpack('<Q', length_bytes)[0])
            manifest['source_header_sha256'] = hashlib.sha256(length_bytes + header_bytes).hexdigest()
        with open_source(source_path) as source:
            count = validate_source(source)
            if count != expected_blocks:
                raise ValueError(f'Expected {expected_blocks} blocks, found {count}')
            manifest['n_blocks'] = count
            for start in range(0, count, blocks_per_piece):
                size = min(blocks_per_piece, count-start)
                shards = namespace['_load_shards'](BlockWindow(source, start, size), size)
                for offset in range(size):
                    verify_reconstruction(source, start+offset, [shards[r][offset] for r in range(2)])
                for rank in range(2):
                    tensors, scalars = flatten(shards[rank], start)
                    filename = f'rank{rank}_blocks_{start:02d}-{start+size-1:02d}.safetensors'
                    path = temporary/filename
                    save_file(tensors, str(path), metadata={'format': 'pt', 'layout_version': '1',
                              'modified': 'H3 block weights repartitioned into TP rank shards; values unchanged',
                              'source_sha256': digest, 'rank': str(rank), 'scalars': json.dumps(scalars, sort_keys=True)})
                    with open(path, 'r+b') as handle:
                        os.fsync(handle.fileno())
                    descriptor = dict(file=filename, rank=rank, block_start=start, block_end=start+size,
                                      file_bytes=path.stat().st_size, sha256=sha256(path), scalars=scalars,
                                      tensors=tensor_schema(tensors), resident_bytes=sum(t.numel()*t.element_size() for t in tensors.values()))
                    readback = load_piece(temporary, descriptor)
                    if any(not equal_bytes(readback[key], value) for key, value in tensors.items()):
                        raise ValueError('Piece round trip differs: ' + filename)
                    manifest['pieces'].append(descriptor)
                    del tensors, readback
                namespace['_SHARD_CACHE'].clear()
                del shards
                gc.collect()
                print(f'Verified blocks {start}-{start+size-1}, both ranks', flush=True)
        after = source_path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Source changed during export')
        # Re-read every completed file before publishing the manifest.
        for piece in manifest['pieces']:
            del_tensors = load_piece(temporary, piece)
            del del_tensors
        manifest['verification'] = dict(all_tensor_roundtrips_bit_exact=True,
                                       all_source_reconstructions_bit_exact=True,
                                       final_piece_hashes_and_schemas_verified=True)
        with open(temporary/'manifest.json', 'w', encoding='utf-8') as handle:
            json.dump(manifest, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.rename(destination)
        return destination, manifest
    except BaseException:
        # Only this invocation's mkdtemp directory is eligible for cleanup.
        gc.collect()
        shutil.rmtree(temporary, ignore_errors=True)
        raise
