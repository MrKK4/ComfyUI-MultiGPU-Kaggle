"""Validated, bounded-memory loading of H3 layout-v1 TP pieces. No network access."""
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import stat
import struct
import time

import torch

logger = logging.getLogger('MultiGPU')
_SOURCE_HASHES = {}
_DTYPES = {'I8': (torch.int8, 1), 'F16': (torch.float16, 2),
           'BF16': (torch.bfloat16, 2), 'F32': (torch.float32, 4)}
_MAX_HEADER = 16 << 20


class PieceFileError(ValueError):
    """A source/artifact validation or read error; no CUDA error is wrapped in this type."""


def _digest(path):
    sha = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            sha.update(block)
    return sha.hexdigest()


def _header(data, file_bytes):
    length = struct.unpack_from('<Q', data)[0]
    if not 1 <= length <= min(_MAX_HEADER, file_bytes - 8):
        raise PieceFileError('Invalid safetensors header length')
    header = json.loads(bytes(data[8:8+length]))
    if not isinstance(header, dict):
        raise PieceFileError('Invalid safetensors header')
    return header, 8+length


def _source_header(path):
    with open(path, 'rb') as handle:
        first = handle.read(8)
        if len(first) != 8:
            raise PieceFileError('Truncated source header')
        length = struct.unpack('<Q', first)[0]
        if not 1 <= length <= _MAX_HEADER:
            raise PieceFileError('Source header is too large')
        raw = first + handle.read(length)
    if len(raw) != length+8:
        raise PieceFileError('Truncated source header')
    header, _ = _header(raw, path.stat().st_size)
    keys = [key for key in header if key != '__metadata__']
    prefix = next((p for p in ('model.diffusion_model.', 'diffusion_model.')
                   if keys and all(key.startswith(p) for key in keys)), '')
    return {key[len(prefix):]: value for key, value in header.items() if key != '__metadata__'}, hashlib.sha256(raw).hexdigest()


def expected_schema(header, start, end):
    tensors, scalars = {}, {}
    for index in range(start, end):
        prefix = f'blocks.{index}.'
        qkv = header[prefix+'attn.qkv_proj.weight']['shape']
        hidden, inner = qkv[1], qkv[0]//3
        ffn = header[prefix+'mlp.fc1.weight']['shape'][0]//2
        if (qkv != [3*inner, hidden] or hidden % 256 or inner % 512 or ffn % 512
                or header[prefix+'attn.out_proj.weight']['shape'] != [hidden, inner]
                or header[prefix+'mlp.fc1.weight']['shape'] != [2*ffn, hidden]
                or header[prefix+'mlp.fc2.weight']['shape'] != [hidden, ffn]):
            raise PieceFileError('Unsupported source dimensions: '+prefix)
        for name, source_name, row_split in (('qkv','attn.qkv_proj',True), ('out','attn.out_proj',False),
                                              ('fc1','mlp.fc1',True), ('fc2','mlp.fc2',False)):
            weight = header[prefix+source_name+'.weight']
            scale = header[prefix+source_name+'.weight_scale']
            if weight['dtype'] != 'I8' or scale['dtype'] not in ('F16','BF16','F32'):
                raise PieceFileError('Unsupported source weight/scale dtype')
            if scale['shape'] not in ([weight['shape'][0]], [weight['shape'][0],1]):
                raise PieceFileError('Expected per-row scales')
            for suffix, entry in (('weight',weight), ('scale',scale)):
                shape = list(entry['shape'])
                if row_split:
                    shape[0] //= 2
                elif suffix == 'weight':
                    shape[1] //= 2
                dtype, itemsize = _DTYPES[entry['dtype']]
                tensors[prefix+name+'.'+suffix] = dict(shape=shape,dtype=str(dtype),nbytes=math.prod(shape)*itemsize)
        for name, source_name in (('norm1','norm1.weight'), ('norm2','norm2.weight'),
                                  ('qn','attn.q_norm.weight'), ('kn','attn.k_norm.weight')):
            entry = header[prefix+source_name]
            code = 'F32' if name.startswith('norm') else entry['dtype']
            if entry['shape'] != ([hidden] if name.startswith('norm') else [128]) or code not in _DTYPES:
                raise PieceFileError('Unsupported source norm')
            dtype, itemsize = _DTYPES[code]
            tensors[prefix+name] = dict(shape=entry['shape'],dtype=str(dtype),nbytes=math.prod(entry['shape'])*itemsize)
        scalars[prefix+'heads'], scalars[prefix+'inner'] = inner//256, inner//2
    return tensors, scalars


def validate_manifest(directory, source_path, n_blocks, manifest_sha256):
    """Pin the manifest, verify the whole source, and validate every descriptor before allocation."""
    started = time.perf_counter()
    root, source = Path(directory).resolve(), Path(source_path).resolve()
    try:
        if not isinstance(manifest_sha256,str) or not re.fullmatch('[0-9a-f]{64}',manifest_sha256):
            raise PieceFileError('A pinned MMH3_TP_PIECES_MANIFEST_SHA256 is required')
        raw = (root/'manifest.json').read_bytes()
        if len(raw) > 8 << 20 or hashlib.sha256(raw).hexdigest() != manifest_sha256:
            raise PieceFileError('Manifest SHA-256 mismatch')
        manifest = json.loads(raw)
        for key, value in (('layout_version',1),('tp_degree',2),('head_dim',128),('convrot_group_size',256),('n_blocks',n_blocks)):
            if type(manifest.get(key)) is not int or manifest[key] != value:
                raise PieceFileError('Unsupported manifest '+key)
        grouping = manifest['blocks_per_piece']
        if type(grouping) is not int or grouping < 1 or grouping > n_blocks:
            raise PieceFileError('Invalid piece grouping')
        before = source.stat()
        if before.st_size != manifest['source']['size']:
            raise PieceFileError('Source size mismatch')
        header, header_sha = _source_header(source)
        if header_sha != manifest['source_header_sha256']:
            raise PieceFileError('Source header hash mismatch')
        indices = {int(m[1]) for key in header if (m := re.match(r'^blocks\.(\d+)\.',key))}
        if indices != set(range(n_blocks)):
            raise PieceFileError('Source block count mismatch')
        fingerprint = (str(source),before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)
        source_started = time.perf_counter()
        source_sha = _SOURCE_HASHES.get(fingerprint)
        source_cached = source_sha is not None
        if source_sha is None:
            source_sha = _digest(source)
            after = source.stat()
            if (before.st_size,before.st_mtime_ns,before.st_ctime_ns) != (after.st_size,after.st_mtime_ns,after.st_ctime_ns):
                raise PieceFileError('Source changed during validation')
            _SOURCE_HASHES.clear()
            _SOURCE_HASHES[fingerprint] = source_sha
        if source_sha != manifest['source']['sha256']:
            raise PieceFileError('Source SHA-256 mismatch')
        source_seconds = time.perf_counter()-source_started
        pieces = manifest['pieces']
        ranges = [(i,min(i+grouping,n_blocks)) for i in range(0,n_blocks,grouping)]
        if len(pieces) != 2*len(ranges):
            raise PieceFileError('Piece count mismatch')
        for rank in range(2):
            rank_pieces = sorted((p for p in pieces if type(p['rank']) is int and p['rank']==rank),key=lambda p:p['block_start'])
            if [(p['block_start'],p['block_end']) for p in rank_pieces] != ranges:
                raise PieceFileError('Piece block coverage mismatch')
            for piece in rank_pieces:
                start,end = piece['block_start'],piece['block_end']
                if type(start) is not int or type(end) is not int:
                    raise PieceFileError('Invalid block range')
                filename = f'rank{rank}_blocks_{start:02d}-{end-1:02d}.safetensors'
                if piece['file'] != filename:
                    raise PieceFileError('Unsafe or inconsistent piece filename')
                path = root/filename
                if path.is_symlink() or not path.resolve().is_relative_to(root) or not stat.S_ISREG(path.stat().st_mode):
                    raise PieceFileError('Piece is not a regular contained file')
                if type(piece['file_bytes']) is not int or path.stat().st_size != piece['file_bytes']:
                    raise PieceFileError('Piece file size mismatch: '+filename)
                if not re.fullmatch('[0-9a-f]{64}',piece['sha256']):
                    raise PieceFileError('Invalid piece SHA-256')
                tensors,scalars = expected_schema(header,start,end)
                if piece['tensors'] != tensors or piece['scalars'] != scalars:
                    raise PieceFileError('Piece schema/scalars disagree with source header: '+filename)
                if piece['resident_bytes'] != sum(t['nbytes'] for t in tensors.values()):
                    raise PieceFileError('Piece resident byte count mismatch')
        manifest['_validation_seconds'] = time.perf_counter()-started
        manifest['_source_sha_seconds'] = source_seconds
        manifest['_source_sha_cached'] = source_cached
        return root,manifest
    except PieceFileError:
        raise
    except (OSError,ValueError,KeyError,TypeError,IndexError,struct.error) as error:
        raise PieceFileError(str(error)) from None


def read_piece_into(buffer, root, piece, source_sha):
    """One read+hash pass into a reusable buffer, all validation before H2D."""
    try:
        view = memoryview(buffer.numpy())[:piece['file_bytes']]
        digest = hashlib.sha256()
        with open(root/piece['file'],'rb',buffering=0) as handle:
            cursor = 0
            while cursor < len(view):
                count = handle.readinto(view[cursor:min(cursor+(8<<20),len(view))])
                if not count:
                    raise PieceFileError('Truncated piece: '+piece['file'])
                digest.update(view[cursor:cursor+count])
                cursor += count
            if handle.read(1):
                raise PieceFileError('Extended piece: '+piece['file'])
        if digest.hexdigest() != piece['sha256']:
            raise PieceFileError('Piece SHA-256 mismatch: '+piece['file'])
        header,data_start = _header(view,len(view))
        metadata = header.pop('__metadata__',{})
        if (metadata.get('layout_version') != '1' or metadata.get('rank') != str(piece['rank'])
                or metadata.get('source_sha256') != source_sha
                or json.loads(metadata.get('scalars','{}')) != piece['scalars']):
            raise PieceFileError('Piece metadata mismatch')
        schema,intervals = {},[]
        for name, entry in header.items():
            code, shape, offsets = entry['dtype'],entry['shape'],entry['data_offsets']
            if (code not in _DTYPES or not isinstance(shape,list) or any(type(x) is not int or x < 1 for x in shape)
                    or not isinstance(offsets,list) or len(offsets)!=2 or any(type(x) is not int for x in offsets)):
                raise PieceFileError('Invalid tensor record')
            dtype,itemsize = _DTYPES[code]
            start,end = offsets
            nbytes = math.prod(shape)*itemsize
            if start < 0 or end-start != nbytes:
                raise PieceFileError('Invalid tensor offsets/length')
            intervals.append((start,end))
            schema[name] = dict(shape=shape,dtype=str(dtype),nbytes=nbytes)
        if schema != piece['tensors']:
            raise PieceFileError('Piece tensor schema mismatch')
        cursor = 0
        for start,end in sorted(intervals):
            if start != cursor:
                raise PieceFileError('Overlapping or non-contiguous tensor offsets')
            cursor = end
        if cursor != piece['file_bytes']-data_start:
            raise PieceFileError('Tensor payload extent mismatch')
        return header,data_start
    except PieceFileError:
        raise
    except (OSError,ValueError,KeyError,TypeError,IndexError,struct.error) as error:
        raise PieceFileError(str(error)) from None


def _restore_rank(rank, device, pieces, root, source_sha, n_blocks):
    blocks = [None]*n_blocks
    timings = dict(rank=rank,pieces=len(pieces),read_hash_seconds=0.,h2d_seconds=0.)
    buffer = torch.empty(max(p['file_bytes'] for p in pieces),dtype=torch.uint8,pin_memory=device.type=='cuda')
    converted = {}
    for piece in pieces:
        started = time.perf_counter()
        try:
            header,data_start = read_piece_into(buffer,root,piece,source_sha)
        except PieceFileError as error:
            message = str(error)
            blocks.clear()
            converted.clear()
            return None,message,timings
        timings['read_hash_seconds'] += time.perf_counter()-started
        started = time.perf_counter()
        for name,entry in header.items():
            a,b = entry['data_offsets']
            # Copy bytes first: typed views of a file offset may be unaligned.
            converted[name] = buffer[data_start+a:data_start+b].to(device,non_blocking=True,copy=True).view(
                _DTYPES[entry['dtype']][0]).reshape(entry['shape'])
        if device.type == 'cuda':
            torch.cuda.current_stream(device).synchronize()
        timings['h2d_seconds'] += time.perf_counter()-started
        for index in range(piece['block_start'],piece['block_end']):
            prefix = f'blocks.{index}.'
            block = dict(heads=piece['scalars'][prefix+'heads'],inner=piece['scalars'][prefix+'inner'])
            for name in ('qkv','out','fc1','fc2'):
                block[name] = (converted[prefix+name+'.weight'],converted[prefix+name+'.scale'])
            for name in ('norm1','norm2','qn','kn'):
                block[name] = converted[prefix+name]
            blocks[index] = block
        converted.clear()
    return blocks,None,timings


def restore_ranks(source_path, n_blocks, directory, manifest_sha256, cached, devices, free_memory, get_free_memory):
    started = time.perf_counter()
    root,manifest = validate_manifest(directory,source_path,n_blocks,manifest_sha256)
    missing = [rank for rank in range(2) if cached is None or cached[rank] is None]
    pieces = {rank:sorted((p for p in manifest['pieces'] if p['rank']==rank),key=lambda p:p['block_start']) for rank in missing}
    for rank in missing:
        needed = sum(p['resident_bytes'] for p in pieces[rank])+(2<<30)
        free_memory(needed,devices[rank])
        available = get_free_memory(devices[rank])
        if available < needed:
            raise RuntimeError(f'[TP-PIECES] cuda rank {rank}: need {needed/2**30:.2f} GiB free, have {available/2**30:.2f} GiB; free other models or reduce workload')

    # Inference mode is thread-local. Match the original builder's tensor
    # dispatch keys so compiled norm kernels can reuse the same variants.
    caller_inference_mode = torch.is_inference_mode_enabled()

    @torch.inference_mode(caller_inference_mode)
    def worker(rank):
        device = torch.device(devices[rank])
        if device.type == 'cuda':
            with torch.cuda.device(device):
                return _restore_rank(rank,device,pieces[rank],root,manifest['source']['sha256'],n_blocks)
        return _restore_rank(rank,device,pieces[rank],root,manifest['source']['sha256'],n_blocks)

    completed,failures = [],[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker,rank) for rank in missing]
        for rank,future in zip(missing,futures):
            try:
                completed.append((rank,*future.result()))
            except BaseException as error:
                failures.append(error)
    # All workers have stopped. No cache reference has been changed yet.
    file_errors = [error for _,_,error,_ in completed if error]
    if failures or file_errors:
        for _,blocks,_,_ in completed:
            if blocks is not None:
                blocks.clear()
        completed.clear()
        futures.clear()
        gc.collect()
        if failures:
            raise failures[0]
        for rank in missing:
            if torch.device(devices[rank]).type == 'cuda':
                with torch.cuda.device(devices[rank]):
                    torch.cuda.synchronize(devices[rank])
                    torch.cuda.empty_cache()
        raise PieceFileError('; '.join(file_errors))
    shards = list(cached) if cached is not None else [None,None]
    for rank,blocks,_,timings in completed:
        shards[rank] = blocks
        logger.info('[TP-PIECES] rank=%d files=%d read_hash=%.3fs h2d=%.3fs',
                    rank,timings['pieces'],timings['read_hash_seconds'],timings['h2d_seconds'])
    logger.info('[TP-PIECES] loaded ranks=%s total=%.3fs validation=%.3fs source_sha=%.3fs source_cached=%s pinned_peak=%.3fGiB',
                missing,time.perf_counter()-started,manifest['_validation_seconds'],manifest['_source_sha_seconds'],
                manifest['_source_sha_cached'],sum(max(p['file_bytes'] for p in pieces[r]) for r in missing)/2**30)
    # Worker-local buffers have been dropped. PyTorch may cache their bounded pinned allocations.
    return shards
