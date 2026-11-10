"""CPU checks against the actual H3 splitter and independent inverse operations."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from safetensors.torch import save_file
import torch

from portkit.h3_pieces import (export_pieces, flatten, load_piece, open_source,
                               reference_builder, sha256, validate_source, equal_bytes)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/'h3_tensor_parallel.py'


def fixture(path):
    data = {}
    for block in range(4):
        prefix = f'model.diffusion_model.blocks.{block}.'
        for name, rows, columns in (('attn.qkv_proj',1536,512), ('attn.out_proj',512,512),
                                    ('mlp.fc1',2048,512), ('mlp.fc2',512,1024)):
            r, c = torch.arange(rows)[:,None], torch.arange(columns)[None,:]
            data[prefix+name+'.weight'] = (((r//128)*17 + c%127 + block*7)%255-128).to(torch.int8)
            data[prefix+name+'.weight_scale'] = (torch.arange(rows).float()/1000+block+1).reshape(-1,1)
            config = json.dumps(dict(format='int8_tensorwise',convrot=True,convrot_groupsize=256))
            data[prefix+name+'.comfy_quant'] = torch.tensor(list(config.encode()), dtype=torch.uint8)
        for key, count in (('norm1.weight',512), ('norm2.weight',512),
                           ('attn.q_norm.weight',128), ('attn.k_norm.weight',128)):
            data[prefix+key] = (torch.arange(count).float()/1000+block).half()
    save_file(data,str(path))


class PieceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        scratch = ROOT/'portkit_runs/h3_pieces_export/test_tmp'
        scratch.mkdir(parents=True,exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=scratch)
        cls.root = Path(cls.temp.name)
        cls.source = cls.root/'source.safetensors'
        fixture(cls.source)
        cls.path, cls.manifest = export_pieces(cls.source, cls.root/'export', REFERENCE,
                                             {'sha256': sha256(cls.source)}, expected_blocks=4)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_every_exported_tensor_matches_unwindowed_original_splitter(self):
        namespace, _ = reference_builder(REFERENCE)
        with open_source(self.source) as source:
            self.assertEqual(validate_source(source),4)
            # Full original splitter is an independent test of the windowed export assembly.
            expected = namespace['_load_shards'](source,4)
            for piece in self.manifest['pieces']:
                tensors = load_piece(self.path,piece)
                start, end = piece['block_start'], piece['block_end']
                original, scalars = flatten(expected[piece['rank']][start:end],start)
                self.assertEqual(piece['scalars'],scalars)
                self.assertEqual(set(tensors),set(original))
                for key in tensors:
                    self.assertTrue(equal_bytes(tensors[key],original[key]),key)
        self.assertEqual(len(self.manifest['pieces']),4)
        self.assertTrue(all(self.manifest['verification'].values()))
        self.assertTrue((self.path/'manifest.json').exists())
        self.assertFalse(self.manifest['gpu_consumer_validated'])

    def test_corruption_is_refused(self):
        piece = self.manifest['pieces'][0]
        target = self.path/piece['file']
        original = target.read_bytes()
        try:
            altered = bytearray(original)
            altered[-1] ^= 1
            target.write_bytes(altered)
            with self.assertRaisesRegex(ValueError,'SHA-256'):
                load_piece(self.path,piece)
        finally:
            target.write_bytes(original)

    def test_manifest_metadata_and_path_mismatches_are_refused(self):
        piece = copy.deepcopy(self.manifest['pieces'][0])
        piece['scalars']['blocks.0.heads'] = 999
        with self.assertRaisesRegex(ValueError,'metadata'):
            load_piece(self.path,piece)
        piece['file'] = '../outside.safetensors'
        with self.assertRaisesRegex(ValueError,'filename'):
            load_piece(self.path,piece)

    def test_wrong_source_identity_is_refused_before_export(self):
        with self.assertRaisesRegex(ValueError,'Source file SHA-256'):
            export_pieces(self.source,self.root/'wrong',REFERENCE,{'sha256':'0'*64},expected_blocks=4)


if __name__ == '__main__':
    unittest.main()
