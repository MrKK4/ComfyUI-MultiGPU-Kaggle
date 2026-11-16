"""Stacked LoRAs: every adapter on a split projection must reach the output (sum of all, on both ranks)."""
import ast
from pathlib import Path
import types
import unittest

import torch

tree = ast.parse((Path(__file__).resolve().parents[1] / 'h3_tensor_parallel.py').read_text())


class Adapter:
    def __init__(self, up, down, alpha):
        self.weights = (up, down, alpha, None, None, None)


ns = {'torch': torch, 'comfy': types.SimpleNamespace(weight_adapter=types.SimpleNamespace(LoRAAdapter=Adapter))}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                              and n.name in ('_rows', '_lora_factors', '_shard_lora', '_lora')], type_ignores=[]), 'h3-lora', 'exec'), ns)


class StackTests(unittest.TestCase):
    def test_three_stacked_adapters_all_contribute_on_every_projection(self):
        torch.manual_seed(0)
        shapes = {'attn.qkv_proj': (24, 8), 'mlp.fc1': (16, 8), 'attn.out_proj': (8, 8), 'mlp.fc2': (8, 16)}   # (out, in)
        for name, (out, inp) in shapes.items():
            entries, full = [], torch.zeros(out, inp, dtype=torch.float64)
            for rank, alpha, strength in ((2, 4.0, 0.8), (3, None, -0.5), (1, 1.0, 1.3)):
                up, down = torch.randn(out, rank), torch.randn(rank, inp)
                entries.append((strength, Adapter(up, down, alpha), 1.0, None, None))
                full += strength * (alpha / rank if alpha is not None else 1.0) * (up.double() @ down.double())
            down, up = ns['_lora_factors'](entries)
            x = torch.randn(5, inp)
            want = x.double() @ full.T
            if name in ('attn.qkv_proj', 'mlp.fc1'):   # column-parallel: each rank computes its own output rows
                chunk = out // (3 if name == 'attn.qkv_proj' else 2)
                for rank in (0, 1):
                    got = ns['_lora'](x, ns['_shard_lora'](name, down, up, rank))
                    torch.testing.assert_close(got.double(), ns['_rows'](want.T, chunk, chunk // 2, rank).T, rtol=1e-4, atol=1e-4)
            else:                                      # row-parallel: each rank sees half the input, partial sums add up
                half = inp // 2
                got = sum(ns['_lora'](x[:, r * half:(r + 1) * half], ns['_shard_lora'](name, down, up, r)) for r in (0, 1))
                torch.testing.assert_close(got.double(), want, rtol=1e-4, atol=1e-4)
            for skip in range(3):                      # dropping any one adapter must change the result
                d2, u2 = ns['_lora_factors'](entries[:skip] + entries[skip + 1:])
                self.assertFalse(torch.allclose(x @ (u2 @ d2).T, x @ (up @ down).T, atol=1e-3))


if __name__ == '__main__':
    unittest.main()
