import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

from safetensors.torch import load_file, save_file
import torch

from ci.test_h3_pieces import fixture, REFERENCE
from portkit.h3_pieces import export_pieces, sha256
from portkit.h3_bundle import export_bundle, verify_bundle


class BundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        scratch=REFERENCE.parent/'portkit_runs/h3_standalone/test_tmp'
        scratch.mkdir(parents=True,exist_ok=True)
        cls.temp=tempfile.TemporaryDirectory(dir=scratch)
        cls.root=Path(cls.temp.name)
        cls.source=cls.root/'source.safetensors'
        fixture(cls.source)
        tensors=load_file(str(cls.source))
        prefix='model.diffusion_model.'
        for block in range(4):
            tensors[prefix+f'blocks.{block}.adaln_proj.linear.weight']=torch.randn(32,8)
            tensors[prefix+f'blocks.{block}.adaln_proj.linear.bias']=torch.randn(32)
        tensors[prefix+'video_patch_proj.weight']=torch.randn(512,96)
        tensors[prefix+'test_scalar']=torch.tensor(7,dtype=torch.int64)
        tensors[prefix+'test_empty']=torch.empty(0,5)
        tensors[prefix+'test_bool']=torch.tensor(True)
        metadata={'format':'pt','config':json.dumps({'transformer':{'pdd_steps':1}}),'test':'preserve'}
        save_file(tensors,str(cls.source),metadata=metadata)
        cls.pieces, legacy=export_pieces(cls.source,cls.root/'pieces',REFERENCE,expected_blocks=4)
        cls.identity=legacy['source']
        reference=dict(repo='fixture/test',revision='0'*40,path='layout-v1/fixture',
            manifest=dict(file='manifest.json',file_bytes=(cls.pieces/'manifest.json').stat().st_size,
                          sha256=sha256(cls.pieces/'manifest.json')))
        config={'detected_unet_config':{'image_model':'minimax_h3','num_layers':4}}
        cls.bundle,cls.manifest=export_bundle(cls.source,cls.pieces,cls.root/'bundles',
                                              cls.identity,config,reference)
        cls.manifest_bytes=(cls.bundle/'bundle_manifest.json').read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def verify(self):
        return verify_bundle(self.bundle,self.pieces,sha256(self.bundle/'bundle_manifest.json'))

    def test_complete_reconstruction_without_source(self):
        hidden=self.source.with_suffix('.hidden')
        self.source.rename(hidden)
        try:
            result=self.verify()
            self.assertEqual(result['virtual_source_sha256'],self.identity['sha256'])
            self.assertEqual(result['projection_tensor_count'],32)
            companion=load_file(str(self.bundle/'companion.safetensors'))
            self.assertTrue(any('adaln_proj.linear.weight' in key for key in companion))
            self.assertEqual(companion['model.diffusion_model.test_scalar'].ndim,0)
            self.assertEqual(companion['model.diffusion_model.test_empty'].numel(),0)
            self.assertFalse(self.manifest['standalone_loader_validated'])
        finally:
            hidden.rename(self.source)

    def test_runtime_preflight_and_rank_restore_need_no_source_file(self):
        package=types.ModuleType('h3_bundle_runtime_fixture')
        package.__path__=[str(REFERENCE.parent)]
        sys.modules[package.__name__]=package
        name=package.__name__+'.h3_bundle_source'
        spec=importlib.util.spec_from_file_location(name,REFERENCE.parent/'h3_bundle_source.py')
        runtime=importlib.util.module_from_spec(spec)
        sys.modules[name]=runtime
        spec.loader.exec_module(runtime)
        source=runtime.H3BundleSource(self.bundle,self.pieces,sha256(self.bundle/'bundle_manifest.json'))
        hidden=self.source.with_suffix('.hidden')
        self.source.rename(hidden)
        try:
            manifest,companion,config=source.validate()
            self.assertEqual(manifest['source']['sha256'],self.identity['sha256'])
            self.assertEqual(config['detected_unet_config']['num_layers'],4)
            self.assertTrue(companion.is_file())
            from h3_bundle_runtime_fixture.h3_piece_loader import restore_bundle_ranks
            ranks=restore_bundle_ranks(source,4,None,(torch.device('cpu'),)*2,
                lambda *args:None,lambda device:1<<50)
            self.assertEqual([len(rank) for rank in ranks],[4,4])
            retained=restore_bundle_ranks(source,4,[ranks[0],None],(torch.device('cpu'),)*2,
                lambda *args:None,lambda device:1<<50)
            self.assertIs(retained[0],ranks[0])
            self.assertEqual(len(retained[1]),4)
        finally:
            hidden.rename(self.source)

    def test_partial_restore_preserves_shared_lists_and_rejects_corruption(self):
        package=types.ModuleType('h3_bundle_partial_fixture')
        package.__path__=[str(REFERENCE.parent)]
        sys.modules[package.__name__]=package
        name=package.__name__+'.h3_bundle_source'
        spec=importlib.util.spec_from_file_location(name,REFERENCE.parent/'h3_bundle_source.py')
        runtime=importlib.util.module_from_spec(spec)
        sys.modules[name]=runtime
        spec.loader.exec_module(runtime)
        source=runtime.H3BundleSource(self.bundle,self.pieces,sha256(self.bundle/'bundle_manifest.json'))
        from h3_bundle_partial_fixture.h3_piece_loader import restore_bundle_ranks, PieceFileError
        devices=(torch.device('cpu'),)*2
        restore=lambda ranks:restore_bundle_ranks(source,4,ranks,devices,
            lambda *args:None,lambda device:1<<50)
        ranks=restore(None)
        rank_list=ranks[1]
        expected=ranks[1][0]['qkv'][0].clone()
        ranks[1][0]=ranks[1][1]=None
        self.assertIs(restore(ranks),ranks)
        self.assertIs(ranks[1],rank_list)
        self.assertTrue(torch.equal(ranks[1][0]['qkv'][0],expected))
        ranks[1][0]=ranks[1][1]=None
        piece=next(p for p in self.manifest['pieces'] if p['rank']==1 and p['block_start']==0)
        path=self.pieces/piece['file']
        original=path.read_bytes()
        try:
            path.write_bytes(original[:-1]+bytes([original[-1]^1]))
            with self.assertRaisesRegex(PieceFileError,'SHA-256'):
                restore(ranks)
            self.assertIsNone(ranks[1][0])
            self.assertIsNone(ranks[1][1])
            self.assertIsNotNone(ranks[1][2])
        finally:
            path.write_bytes(original)

    def test_pin_partition_header_and_legacy_binding_rejected(self):
        with self.assertRaisesRegex(ValueError,'pin'):
            verify_bundle(self.bundle,self.pieces,'0'*64)
        for problem in ('partition','header','legacy','origin','filename'):
            altered=copy.deepcopy(self.manifest)
            if problem=='partition': altered['partition']['n_tp']+=1
            elif problem=='header': altered['source_header']['file_bytes']-=1
            elif problem=='legacy': altered['pieces_reference']['manifest']['sha256']='0'*64
            elif problem=='origin': altered['original_tensors'][0]['origin']='invalid'
            else: altered['companion']['file']='../companion.safetensors'
            try:
                (self.bundle/'bundle_manifest.json').write_text(json.dumps(altered))
                with self.assertRaises(ValueError): self.verify()
            finally:
                (self.bundle/'bundle_manifest.json').write_bytes(self.manifest_bytes)

    def test_companion_drop_extra_and_byte_change_rejected(self):
        path=self.bundle/'companion.safetensors'
        original=path.read_bytes()
        from safetensors import safe_open
        with safe_open(str(path),framework='pt',device='cpu') as handle:
            metadata=handle.metadata()
        for problem in ('drop','extra','value'):
            tensors=load_file(str(path))
            name=next(iter(tensors))
            if problem=='drop': tensors.pop(name)
            elif problem=='extra': tensors['extra']=torch.tensor(1.)
            else: tensors[name].reshape(-1).view(torch.uint8)[0]^=1
            altered=copy.deepcopy(self.manifest)
            try:
                save_file(tensors,str(path),metadata=metadata)
                altered['companion'].update(file_bytes=path.stat().st_size,sha256=sha256(path))
                (self.bundle/'bundle_manifest.json').write_text(json.dumps(altered))
                with self.assertRaises(ValueError): self.verify()
            finally:
                path.write_bytes(original)
                (self.bundle/'bundle_manifest.json').write_bytes(self.manifest_bytes)

    def test_missing_corrupt_piece_and_overlap_rejected(self):
        piece=self.manifest['pieces'][0]
        path=self.pieces/piece['file']
        original=path.read_bytes()
        hidden=path.with_suffix('.hidden')
        path.rename(hidden)
        try:
            with self.assertRaises((ValueError,OSError)): self.verify()
        finally:
            hidden.rename(path)
        try:
            path.write_bytes(original[:-1]+bytes([original[-1]^1]))
            with self.assertRaises(ValueError): self.verify()
        finally:
            path.write_bytes(original)
        altered=copy.deepcopy(self.manifest)
        altered['pieces'].append(copy.deepcopy(piece))
        try:
            (self.bundle/'bundle_manifest.json').write_text(json.dumps(altered))
            with self.assertRaises(ValueError): self.verify()
        finally:
            (self.bundle/'bundle_manifest.json').write_bytes(self.manifest_bytes)
