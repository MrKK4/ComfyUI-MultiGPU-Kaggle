"""Pinned, local H3 bundle identity and fast preflight. No full checkpoint access."""
import hashlib
import json
from pathlib import Path
import re
import struct

from safetensors import safe_open


class BundleFileError(ValueError):
    pass


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for data in iter(lambda: handle.read(8 << 20), b''):
            digest.update(data)
    return digest.hexdigest()


def _file(root, name):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', name):
        raise BundleFileError('Unsafe H3 bundle filename')
    path = root/name
    if path.is_symlink() or path.resolve().parent != root or not path.is_file():
        raise BundleFileError('Missing or unsafe H3 bundle file: '+name)
    return path


def _checked(root, descriptor):
    path = _file(root, descriptor['file'])
    if path.stat().st_size != descriptor['file_bytes'] or _sha256(path) != descriptor['sha256']:
        raise BundleFileError('H3 bundle file differs: '+path.name)
    return path


class H3BundleSource:
    """Immutable artifact identity. Paths name local files; hashes bind their contents."""
    def __init__(self, bundle_directory, pieces_directory, manifest_sha256):
        self.bundle_directory = Path(bundle_directory).resolve()
        self.pieces_directory = Path(pieces_directory).resolve()
        self.manifest_sha256 = manifest_sha256

    def __eq__(self, other):
        return isinstance(other, H3BundleSource) and self.identity == other.identity

    def __hash__(self):
        return hash(self.identity)

    @property
    def identity(self):
        return (self.bundle_directory, self.pieces_directory, self.manifest_sha256)

    def validate(self):
        """Check pinned metadata and companion; shard file hashes are checked on bounded read."""
        from .h3_piece_loader import expected_schema

        root, pieces_root = self.bundle_directory, self.pieces_directory
        if not isinstance(self.manifest_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', self.manifest_sha256):
            raise BundleFileError('Pinned bundle manifest SHA-256 is required')
        raw = _file(root, 'bundle_manifest.json').read_bytes()
        if len(raw) > 8 << 20 or hashlib.sha256(raw).hexdigest() != self.manifest_sha256:
            raise BundleFileError('H3 bundle manifest SHA-256 mismatch')
        manifest = json.loads(raw)
        if (manifest['bundle_layout_version'] != 1 or manifest['piece_layout_version'] != 1
                or manifest['tp_degree'] != 2 or manifest['head_dim'] != 128
                or manifest['convrot_group_size'] != 256 or manifest['requires_full_checkpoint'] is not False):
            raise BundleFileError('Unsupported H3 bundle format')
        if (type(manifest['n_blocks']) is not int or not 1 <= manifest['n_blocks'] <= 50
                or manifest['key_prefix'] not in ('', 'model.diffusion_model.', 'diffusion_model.')):
            raise BundleFileError('Unsupported H3 block count or prefix')
        legacy_path = _checked(pieces_root, manifest['pieces_reference']['manifest'])
        legacy = json.loads(legacy_path.read_text(encoding='utf-8'))
        if (legacy['layout_version'] != 1 or legacy['source'] != manifest['source']
                or legacy['pieces'] != manifest['pieces'] or legacy['n_blocks'] != manifest['n_blocks']
                or legacy['tp_degree'] != 2):
            raise BundleFileError('H3 piece manifest does not match bundle')
        source_header = _checked(root, manifest['source_header']).read_bytes()
        if (len(source_header) < 8 or struct.unpack_from('<Q', source_header)[0]+8 != len(source_header)
                or hashlib.sha256(source_header).hexdigest() != legacy['source_header_sha256']):
            raise BundleFileError('H3 original header differs')
        header = json.loads(source_header[8:])
        original = {k:v for k,v in header.items() if k != '__metadata__'}
        prefix = manifest['key_prefix']
        if header.get('__metadata__', {}) != manifest['safetensors_metadata']:
            raise BundleFileError('H3 original metadata differs')
        if not original or any(not k.startswith(prefix) for k in original):
            raise BundleFileError('H3 source key prefix differs')
        tp_keys = set(manifest['partition']['tp_keys'])
        source_keys = set(original)
        companion_keys = source_keys-tp_keys
        if (manifest['partition']['n_source'] != len(source_keys)
                or manifest['partition']['n_companion'] != len(companion_keys)
                or manifest['partition']['n_tp'] != len(tp_keys)
                or len(tp_keys) != 8*manifest['n_blocks']
                or len(manifest['tp_modules']) != 4*manifest['n_blocks']):
            raise BundleFileError('H3 tensor partition differs')
        expected_tp = {prefix+f'blocks.{i}.{projection}.{suffix}'
            for i in range(manifest['n_blocks'])
            for projection in ('attn.qkv_proj','attn.out_proj','mlp.fc1','mlp.fc2')
            for suffix in ('weight','weight_scale')}
        if tp_keys != expected_tp or not tp_keys <= source_keys:
            raise BundleFileError('H3 TP key partition differs')
        catalog = manifest['original_tensors']
        if len(catalog) != len(source_keys) or {row['name'] for row in catalog} != source_keys:
            raise BundleFileError('H3 source tensor catalog differs')
        for row in catalog:
            name = row['name']
            if any(row[k] != original[name][k] for k in ('shape','dtype','data_offsets')):
                raise BundleFileError('H3 original tensor schema differs: '+name)
        companion_path = _checked(root, manifest['companion'])
        with safe_open(str(companion_path), framework='pt', device='cpu') as companion:
            if set(companion.keys()) != companion_keys or companion.metadata() != manifest['safetensors_metadata']:
                raise BundleFileError('H3 companion keyset or metadata differs')
            descriptors = manifest['companion']['tensors']
            if set(descriptors) != companion_keys:
                raise BundleFileError('H3 companion descriptor keyset differs')
            for name in companion_keys:
                view = companion.get_slice(name)
                record = descriptors[name]
                if (view.get_shape() != original[name]['shape'] or view.get_dtype() != original[name]['dtype']
                        or record['shape'] != original[name]['shape'] or record['dtype'] != original[name]['dtype']):
                    raise BundleFileError('H3 companion tensor schema differs: '+name)
        config_path = _checked(root, manifest['model_config'])
        config = json.loads(config_path.read_text(encoding='utf-8'))
        if (config['detected_unet_config']['image_model'] != 'minimax_h3'
                or config['detected_unet_config']['num_layers'] != manifest['n_blocks']):
            raise BundleFileError('H3 model configuration differs')
        # Exact coverage, shapes, dtypes and sizes before any piece is sent to CUDA.
        unprefixed = {k[len(prefix):]:v for k,v in original.items()}
        by_rank = {0:[],1:[]}
        for piece in manifest['pieces']:
            rank, start, end = piece['rank'], piece['block_start'], piece['block_end']
            if rank not in by_rank or not 0 <= start < end <= manifest['n_blocks']:
                raise BundleFileError('Invalid H3 piece rank or range')
            if piece['file'] != f'rank{rank}_blocks_{start:02d}-{end-1:02d}.safetensors':
                raise BundleFileError('Invalid H3 piece filename')
            path = _file(pieces_root, piece['file'])
            if path.stat().st_size != piece['file_bytes']:
                raise BundleFileError('H3 piece file size differs: '+piece['file'])
            schema, scalars = expected_schema(unprefixed,start,end)
            if piece['tensors'] != schema or piece['scalars'] != scalars:
                raise BundleFileError('H3 piece schema differs: '+piece['file'])
            by_rank[rank].extend(range(start,end))
        if any(sorted(values) != list(range(manifest['n_blocks'])) for values in by_rank.values()):
            raise BundleFileError('Incomplete or duplicate H3 piece block coverage')
        return manifest, companion_path, config
