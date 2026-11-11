"""Lossless standalone H3 artifact assembly. CPU only; not an inference loader."""
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import tempfile

import safetensors
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from portkit.h3_pieces import equal_bytes, load_piece, sha256

_TP = re.compile(r'blocks\.(\d+)\.(attn\.qkv_proj|attn\.out_proj|mlp\.fc1|mlp\.fc2)\.(weight|weight_scale)')
_KINDS = {'attn.qkv_proj': ('qkv', 3), 'attn.out_proj': ('out', 0),
          'mlp.fc1': ('fc1', 2), 'mlp.fc2': ('fc2', 0)}
_CODES = {torch.uint8:'U8', torch.int8:'I8', torch.int16:'I16', torch.int32:'I32',
          torch.int64:'I64', torch.float16:'F16', torch.bfloat16:'BF16',
          torch.float32:'F32', torch.float64:'F64', torch.bool:'BOOL'}
_SIZES = {'U8':1, 'I8':1, 'I16':2, 'I32':4, 'I64':8, 'F16':2, 'BF16':2,
          'F32':4, 'F64':8, 'BOOL':1}


def _bytes(tensor):
    return memoryview(tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy())


def _schema(tensor):
    return dict(dtype=_CODES[tensor.dtype], shape=list(tensor.shape),
                nbytes=tensor.numel()*tensor.element_size())


def _file(root, name):
    root = Path(root).resolve()
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', name):
        raise ValueError('Unsafe bundle filename')
    path = root/name
    if path.is_symlink() or path.resolve().parent != root or not path.is_file():
        raise ValueError('Missing or unsafe bundle file: '+name)
    return path


def _descriptor(path):
    return dict(file=path.name, file_bytes=path.stat().st_size, sha256=sha256(path))


def _checked_file(root, descriptor):
    path = _file(root, descriptor['file'])
    if path.stat().st_size != descriptor['file_bytes'] or sha256(path) != descriptor['sha256']:
        raise ValueError('Bundle file hash/size mismatch: '+path.name)
    return path


def _header(raw, source_bytes):
    if len(raw) < 8:
        raise ValueError('Truncated source header')
    length = struct.unpack_from('<Q', raw)[0]
    if not 0 < length <= 16*1024**2 or len(raw) != length+8:
        raise ValueError('Source header length mismatch')
    header = json.loads(raw[8:])
    metadata = header.get('__metadata__', {})
    if not isinstance(metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k,v in metadata.items()):
        raise ValueError('Invalid source metadata')
    entries = {k:v for k,v in header.items() if k != '__metadata__'}
    cursor = 0
    for name, entry in sorted(entries.items(), key=lambda pair:(*pair[1]['data_offsets'],pair[0])):
        shape, code = entry['shape'], entry['dtype']
        if code not in _SIZES or not isinstance(shape,list) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError('Unsupported source schema: '+name)
        start,end = entry['data_offsets']
        if start != cursor or end-start != math.prod(shape)*_SIZES[code]:
            raise ValueError('Source offsets overlap or leave a gap: '+name)
        cursor = end
    if len(raw)+cursor != source_bytes:
        raise ValueError('Source payload extent mismatch')
    prefix = next((p for p in ('model.diffusion_model.', 'diffusion_model.')
                   if entries and all(k.startswith(p) for k in entries)), '')
    return header, entries, prefix


def _origin(name, prefix):
    match = _TP.fullmatch(name[len(prefix):])
    if match:
        return dict(block=int(match[1]), projection=match[2], suffix=match[3])
    return 'companion'


def _modules(entries, prefix, companion, blocks):
    result = []
    for index in range(blocks):
        for projection, (kind,groups) in _KINDS.items():
            name = f'blocks.{index}.{projection}'
            full = prefix+name
            quant = companion.get_tensor(full+'.comfy_quant')
            config = json.loads(bytes(quant.reshape(-1).tolist()))
            result.append(dict(module=name, weight_key=full+'.weight', scale_key=full+'.weight_scale',
                weight_shape=entries[full+'.weight']['shape'],
                scale_shape=entries[full+'.weight_scale']['shape'],
                scale_dtype=entries[full+'.weight_scale']['dtype'],
                split='grouped_rows' if groups else 'columns', groups=groups, comfy_quant=config))
    return result


class _ProjectionReader:
    def __init__(self, root, pieces, entries, prefix, companion, source_sha, blocks):
        self.root, self.entries, self.prefix = root, entries, prefix
        self.companion, self.source_sha = companion, source_sha
        self.index, self.cache_key, self.pair = {}, None, None
        for descriptor in pieces:
            rank = descriptor['rank']
            if rank not in (0,1):
                raise ValueError('Invalid piece rank')
            for block in range(descriptor['block_start'],descriptor['block_end']):
                if (rank,block) in self.index:
                    raise ValueError('Duplicate piece coverage')
                self.index[rank,block] = descriptor
        if set(self.index) != {(r,b) for r in (0,1) for b in range(blocks)}:
            raise ValueError('Incomplete piece coverage')

    def read_pair(self, block):
        desc = [self.index[r,block] for r in (0,1)]
        key = tuple(d['file'] for d in desc)
        if self.cache_key == key:
            return self.pair
        self.pair = None
        pair = []
        for descriptor in desc:
            values = load_piece(self.root,descriptor)
            with safe_open(str(_file(self.root,descriptor['file'])),framework='pt',device='cpu') as handle:
                if handle.metadata().get('source_sha256') != self.source_sha:
                    raise ValueError('Piece source identity mismatch')
            pair.append(values)
        if (desc[0]['block_start'],desc[0]['block_end']) != (desc[1]['block_start'],desc[1]['block_end']):
            raise ValueError('Piece rank ranges differ')
        for index in range(desc[0]['block_start'],desc[0]['block_end']):
            base = f'blocks.{index}.'
            inner = self.entries[self.prefix+base+'attn.qkv_proj.weight']['shape'][0]//3
            for rank, values in enumerate(pair):
                for key, original in (('norm1','norm1.weight'),('norm2','norm2.weight'),
                                      ('qn','attn.q_norm.weight'),('kn','attn.k_norm.weight')):
                    expected = self.companion.get_tensor(self.prefix+base+original)
                    if key.startswith('norm'):
                        expected = expected.float()
                    if not equal_bytes(values[base+key],expected):
                        raise ValueError('Piece/companion duplicate differs: '+base+key)
                if (desc[rank]['scalars'][base+'inner'] != inner//2
                        or desc[rank]['scalars'][base+'heads'] != inner//256):
                    raise ValueError('Piece scalar differs')
        self.cache_key, self.pair = key, pair
        return pair

    def tensor(self, origin):
        pair = self.read_pair(origin['block'])
        kind,groups = _KINDS[origin['projection']]
        suffix = 'weight' if origin['suffix']=='weight' else 'scale'
        key = f'blocks.{origin["block"]}.{kind}.{suffix}'
        left,right = pair[0][key],pair[1][key]
        if groups:
            if left.shape[0] % groups:
                raise ValueError('Invalid grouped split')
            size = left.shape[0]//groups
            return torch.cat([part for g in range(groups) for part in
                              (left[g*size:(g+1)*size],right[g*size:(g+1)*size])],dim=0)
        if suffix=='weight':
            return torch.cat((left,right),dim=1)
        if not equal_bytes(left,right):
            raise ValueError('Replicated scales differ')
        return left


def verify_bundle(directory, pieces_directory, manifest_sha256):
    """Reconstruct the original SHA using only bundle and piece files. No source argument."""
    manifest_path = _file(directory,'bundle_manifest.json')
    if not re.fullmatch(r'[0-9a-f]{64}',manifest_sha256) or sha256(manifest_path) != manifest_sha256:
        raise ValueError('Bundle manifest pin mismatch')
    manifest = json.loads(manifest_path.read_text())
    if manifest['bundle_layout_version'] != 1 or manifest['piece_layout_version'] != 1:
        raise ValueError('Unsupported bundle layout')
    legacy_path = _checked_file(pieces_directory,manifest['pieces_reference']['manifest'])
    legacy = json.loads(legacy_path.read_text())
    if (legacy['pieces'] != manifest['pieces'] or legacy['source'] != manifest['source']
            or legacy['n_blocks'] != manifest['n_blocks'] or legacy['tp_degree'] != 2):
        raise ValueError('Legacy piece manifest binding differs')
    raw = _checked_file(directory,manifest['source_header']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != legacy['source_header_sha256']:
        raise ValueError('Source header identity differs')
    header,entries,prefix = _header(raw,manifest['source']['size'])
    if prefix != manifest['key_prefix'] or header.get('__metadata__',{}) != manifest['safetensors_metadata']:
        raise ValueError('Original metadata/prefix differs')
    expected_tp = {k for k in entries if _origin(k,prefix) != 'companion'}
    expected_companion = set(entries)-expected_tp
    if (manifest['partition'] != dict(n_source=len(entries),n_companion=len(expected_companion),
            n_tp=len(expected_tp),tp_keys=sorted(expected_tp)) or len(expected_tp) != 8*manifest['n_blocks']):
        raise ValueError('Original tensor partition differs')
    config = json.loads(_checked_file(directory,manifest['model_config']).read_text())
    if (config['detected_unet_config']['num_layers'] != manifest['n_blocks']
            or config['detected_unet_config']['image_model'] != 'minimax_h3'):
        raise ValueError('Model configuration differs')
    for descriptor in manifest['auxiliary_files']:
        _checked_file(directory,descriptor)
    companion_path = _checked_file(directory,manifest['companion'])
    original_rows = manifest['original_tensors']
    ordered = sorted(entries,key=lambda k:(*entries[k]['data_offsets'],k))
    if [row['name'] for row in original_rows] != ordered:
        raise ValueError('Original tensor catalog differs')
    whole = hashlib.sha256(raw)
    with safe_open(str(companion_path),framework='pt',device='cpu') as companion:
        if set(companion.keys()) != expected_companion or companion.metadata() != manifest['safetensors_metadata']:
            raise ValueError('Companion keys or metadata differ')
        if set(manifest['companion']['tensors']) != expected_companion:
            raise ValueError('Companion schema keyset differs')
        if _modules(entries,prefix,companion,manifest['n_blocks']) != manifest['tp_modules']:
            raise ValueError('TP module declaration differs')
        reader = _ProjectionReader(pieces_directory,manifest['pieces'],entries,prefix,companion,
                                   manifest['source']['sha256'],manifest['n_blocks'])
        for row in original_rows:
            name = row['name']
            origin = _origin(name,prefix)
            if row['origin'] != origin or any(row[k] != entries[name][k] for k in ('shape','dtype','data_offsets')):
                raise ValueError('Original tensor origin/schema differs: '+name)
            tensor = companion.get_tensor(name) if origin=='companion' else reader.tensor(origin)
            raw_tensor = _bytes(tensor)
            schema = _schema(tensor)
            digest = hashlib.sha256(raw_tensor).hexdigest()
            expected_schema = dict(dtype=entries[name]['dtype'],shape=entries[name]['shape'],
                nbytes=entries[name]['data_offsets'][1]-entries[name]['data_offsets'][0])
            if schema != expected_schema or digest != row['sha256']:
                raise ValueError('Reconstructed tensor differs: '+name)
            if origin=='companion' and manifest['companion']['tensors'][name] != dict(**schema,sha256=digest):
                raise ValueError('Companion tensor descriptor differs: '+name)
            whole.update(raw_tensor)
    if whole.hexdigest() != manifest['source']['sha256']:
        raise ValueError('Virtual source SHA-256 mismatch')
    return dict(virtual_source_sha256=whole.hexdigest(),source_tensor_count=len(entries),
                companion_tensor_count=len(expected_companion),projection_tensor_count=len(expected_tp))


def export_bundle(source_path, pieces_directory, output_root, identity, config, pieces_reference):
    source_path, output_root = Path(source_path),Path(output_root)
    output_root.mkdir(parents=True,exist_ok=True)
    if source_path.stat().st_size != identity['size'] or sha256(source_path) != identity['sha256']:
        raise ValueError('Original source identity mismatch')
    legacy_path = _checked_file(pieces_directory,pieces_reference['manifest'])
    legacy = json.loads(legacy_path.read_text())
    if legacy['source'] != identity or legacy['layout_version'] != 1 or legacy['tp_degree'] != 2:
        raise ValueError('Source/piece identity differs')
    before = source_path.stat()
    with source_path.open('rb') as handle:
        first=handle.read(8); length=struct.unpack('<Q',first)[0]
        if length > 16*1024**2:
            raise ValueError('Source header too large')
        raw=first+handle.read(length)
    header,entries,prefix = _header(raw,identity['size'])
    if hashlib.sha256(raw).hexdigest() != legacy['source_header_sha256']:
        raise ValueError('Source header does not bind to pieces')
    temporary = Path(tempfile.mkdtemp(prefix='.h3-bundle-',dir=output_root))
    try:
        with safe_open(str(source_path),framework='pt',device='cpu') as source:
            companion = {k:source.get_tensor(k).clone() for k in entries if _origin(k,prefix)=='companion'}
            save_file(companion,str(temporary/'companion.safetensors'),metadata=header.get('__metadata__',{}))
            companion_schema = {k:dict(**_schema(t),sha256=hashlib.sha256(_bytes(t)).hexdigest()) for k,t in companion.items()}
            modules = _modules(entries,prefix,source,legacy['n_blocks'])
            catalog = [dict(name=k,**entries[k],origin=_origin(k,prefix),
                           sha256=hashlib.sha256(_bytes(source.get_tensor(k))).hexdigest())
                       for k in sorted(entries,key=lambda k:(*entries[k]['data_offsets'],k))]
        del companion
        (temporary/'source_header.bin').write_bytes(raw)
        (temporary/'model_config.json').write_text(json.dumps(config,indent=2)+'\n')
        for name in ('LICENSE','NOTICE','source-model-card.md'):
            original=Path(pieces_directory)/name
            if original.is_file():
                shutil.copyfile(_file(pieces_directory,name),temporary/name)
        (temporary/'README.md').write_text(
            '# H3 standalone bundle candidate\n\n'
            'Companion weights plus the pinned existing TP pieces encode the complete original checkpoint.\n'
            'No full checkpoint is required for the reconstruction verifier.\n'
            'This companion file is not a vanilla ComfyUI checkpoint. A bundle-aware inference loader is still required.\n'
            'Standalone GPU inference has not been validated. Existing TP-piece GPU tests are separate.\n'
            'The legacy pieces are immutable and referenced separately; they are not copied here.\n')
        tp_keys=sorted(k for k in entries if _origin(k,prefix)!='companion')
        manifest=dict(bundle_layout_version=1,piece_layout_version=1,source=identity,
            key_prefix=prefix,n_blocks=legacy['n_blocks'],tp_degree=2,head_dim=128,convrot_group_size=256,
            source_header=_descriptor(temporary/'source_header.bin'),
            companion=dict(**_descriptor(temporary/'companion.safetensors'),tensors=companion_schema),
            model_config=_descriptor(temporary/'model_config.json'),pieces_reference=pieces_reference,
            pieces=legacy['pieces'],safetensors_metadata=header.get('__metadata__',{}),
            original_tensors=catalog,tp_modules=modules,
            partition=dict(n_source=len(entries),n_companion=len(companion_schema),n_tp=len(tp_keys),tp_keys=tp_keys),
            exporter_sha256=sha256(__file__),torch_version=torch.__version__,safetensors_version=safetensors.__version__,
            requires_full_checkpoint=False,standalone_loader_validated=False,gpu_consumer_validated=False,
            auxiliary_files=[_descriptor(p) for p in sorted(temporary.iterdir())
                             if p.name in ('LICENSE','NOTICE','source-model-card.md','README.md')])
        manifest_path=temporary/'bundle_manifest.json'
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
        verified=verify_bundle(temporary,pieces_directory,sha256(manifest_path))
        if (before.st_size,before.st_mtime_ns) != (source_path.stat().st_size,source_path.stat().st_mtime_ns):
            raise ValueError('Source changed during export')
        manifest['verification']=dict(**verified,companion_roundtrip_and_projection_bytes_exact=True)
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
        # The final manifest is pinned again after adding the completed proof.
        verify_bundle(temporary,pieces_directory,sha256(manifest_path))
        destination=output_root/(identity['sha256'][:16]+'-'+manifest['companion']['sha256'][:12])
        if destination.exists():
            raise FileExistsError(destination)
        temporary.rename(destination)
        return destination,manifest
    except BaseException:
        shutil.rmtree(temporary,ignore_errors=True)
        raise
