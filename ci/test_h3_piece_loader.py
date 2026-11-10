import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import types
import unittest
from unittest.mock import patch

import torch

import h3_piece_loader as loader
from ci.test_h3_pieces import PieceTests, REFERENCE
from portkit.h3_pieces import flatten, open_source, reference_builder, equal_bytes


class LoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        PieceTests.setUpClass()
        cls.root,cls.source,cls.path = PieceTests.root,PieceTests.source,PieceTests.path
        cls.manifest = PieceTests.manifest
        cls.raw_manifest = (cls.path/'manifest.json').read_bytes()

    @classmethod
    def tearDownClass(cls):
        PieceTests.tearDownClass()

    def pin(self):
        return hashlib.sha256((self.path/'manifest.json').read_bytes()).hexdigest()

    def load(self,cached=None,**kwargs):
        return loader.restore_ranks(str(self.source),4,str(self.path),self.pin(),cached,
                    kwargs.get('devices',(torch.device('cpu'),)*2),lambda *args:None,
                    kwargs.get('free',lambda device:1<<50))

    def compare(self,ranks):
        namespace,_ = reference_builder(REFERENCE)
        with open_source(self.source) as source:
            expected = namespace['_load_shards'](source,4)
        for rank in range(2):
            actual,scalars = flatten(ranks[rank],0)
            want,want_scalars = flatten(expected[rank],0)
            self.assertEqual(scalars,want_scalars)
            self.assertEqual(set(actual),set(want))
            for name in actual:
                self.assertTrue(equal_bytes(actual[name],want[name]),name)

    def test_full_load_and_retained_rank_restore_equal_original(self):
        ranks=self.load()
        self.compare(ranks)
        cached=[ranks[0],None]
        restored=self.load(cached)
        self.assertIs(restored[0],cached[0])
        self.assertIsNone(cached[1])
        self.compare(restored)

    def test_worker_preserves_callers_inference_mode(self):
        # ComfyUI enters inference mode on its execution thread. The rank
        # workers must create the same tensor kind as the original builder.
        for enabled in (True, False):
            with self.subTest(inference=enabled), torch.inference_mode(enabled):
                ranks = self.load()
                for rank in ranks:
                    for block in rank:
                        for value in block.values():
                            for tensor in value if isinstance(value, tuple) else (value,):
                                if torch.is_tensor(tensor):
                                    self.assertEqual(torch.is_inference(tensor), enabled)
                restored = self.load([ranks[0], None])
                self.assertIs(restored[0], ranks[0])
                self.assertEqual(torch.is_inference(restored[1][0]['norm1']), enabled)

    def test_missing_pin_and_manifest_mismatch_are_rejected(self):
        for pin in ('','0'*64):
            with self.assertRaises(loader.PieceFileError):
                loader.validate_manifest(self.path,self.source,4,pin)

    def test_coverage_path_and_scalar_mismatch_are_rejected(self):
        for change in ('coverage','path','scalar'):
            altered=copy.deepcopy(self.manifest)
            if change=='coverage': altered['pieces'].pop()
            if change=='path': altered['pieces'][0]['file']='../outside.safetensors'
            if change=='scalar': altered['pieces'][0]['scalars']['blocks.0.heads']=99
            try:
                (self.path/'manifest.json').write_text(json.dumps(altered))
                with self.assertRaises(loader.PieceFileError):
                    self.load()
            finally:
                (self.path/'manifest.json').write_bytes(self.raw_manifest)

    def test_corrupted_piece_and_midway_read_leave_cached_rank_untouched(self):
        ranks=self.load()
        cached=[ranks[0],None]
        target=self.path/self.manifest['pieces'][1]['file']
        raw=target.read_bytes()
        try:
            target.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
            with self.assertRaisesRegex(loader.PieceFileError,'SHA-256'):
                self.load(cached)
        finally:
            target.write_bytes(raw)
        original=loader.read_piece_into
        def fail(buffer,root,piece,sha):
            if piece['rank']==1 and piece['block_start']==2:
                raise loader.PieceFileError('injected midway read failure')
            return original(buffer,root,piece,sha)
        with patch.object(loader,'read_piece_into',fail):
            with self.assertRaisesRegex(loader.PieceFileError,'midway'):
                self.load(cached)
        self.assertIs(cached[0],ranks[0])
        self.assertIsNone(cached[1])

    def test_middle_source_change_is_detected_even_with_identical_header(self):
        self.load()  # Populate the stat-keyed source hash cache.
        raw=self.source.read_bytes()
        header_length=struct.unpack_from('<Q',raw)[0]
        header=json.loads(raw[8:8+header_length])
        position=8+header_length+header['model.diffusion_model.blocks.1.attn.qkv_proj.weight']['data_offsets'][0]
        altered=bytearray(raw); altered[position]^=1
        try:
            self.source.write_bytes(altered)
            with self.assertRaisesRegex(loader.PieceFileError,'Source SHA-256'):
                self.load()
        finally:
            self.source.write_bytes(raw)

    def test_schema_and_overlapping_offsets_rejected_even_with_updated_hashes(self):
        piece=self.manifest['pieces'][0]
        target=self.path/piece['file']
        raw=target.read_bytes()
        size=struct.unpack_from('<Q',raw)[0]
        for mode in ('shape','offsets'):
            header=json.loads(raw[8:8+size])
            if mode=='shape': header['blocks.0.qkv.weight']['shape'].reverse()
            else: header['blocks.0.norm2']['data_offsets']=header['blocks.0.norm1']['data_offsets']
            packed=json.dumps(header,separators=(',',':')).encode()
            self.assertLessEqual(len(packed),size)
            changed=raw[:8]+packed.ljust(size,b' ')+raw[8+size:]
            manifest=copy.deepcopy(self.manifest)
            manifest['pieces'][0]['sha256']=hashlib.sha256(changed).hexdigest()
            try:
                target.write_bytes(changed)
                (self.path/'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaises(loader.PieceFileError):
                    self.load()
            finally:
                target.write_bytes(raw)
                (self.path/'manifest.json').write_bytes(self.raw_manifest)

    def test_memory_shortfall_and_gpu_errors_are_not_file_fallbacks(self):
        with patch.object(loader,'_restore_rank') as worker:
            with self.assertRaisesRegex(RuntimeError,'need .*GiB'):
                self.load(devices=(torch.device('cuda:0'),torch.device('cuda:1')),free=lambda device:0)
            worker.assert_not_called()
        with patch.object(loader,'_restore_rank',side_effect=RuntimeError('injected CUDA failure')):
            with self.assertRaisesRegex(RuntimeError,'CUDA failure'):
                self.load()

    def test_runtime_hook_defaults_off_and_falls_back_before_sampling(self):
        namespace,_=reference_builder(REFERENCE)
        namespace['os']=__import__('os')
        namespace['__package__']='h3_piece_test_package'
        namespace['open_weights']=lambda path:open_source(path)
        package=types.ModuleType('h3_piece_test_package'); package.__path__=[]
        namespace['comfy'].model_management.get_free_memory=lambda device:1<<50
        modules={'h3_piece_test_package':package,'h3_piece_test_package.h3_piece_loader':loader}
        with patch.dict(sys.modules,modules):
            with patch.dict('os.environ',{'MMH3_TP_PIECES':'0'}):
                baseline=namespace['_load_shards'](str(self.source),4)
            namespace['_SHARD_CACHE'].clear()
            with patch.dict('os.environ',{'MMH3_TP_PIECES':'1','MMH3_TP_PIECES_DIR':str(self.root/'absent')}):
                with self.assertLogs('h3-export',level='WARNING') as logs:
                    fallback=namespace['_load_shards'](str(self.source),4)
            self.assertIn('rebuilding from original',str(logs.output))
        for rank in range(2):
            a,_=flatten(baseline[rank],0); b,_=flatten(fallback[rank],0)
            self.assertTrue(all(equal_bytes(a[k],b[k]) for k in a))


if __name__=='__main__':
    unittest.main()
