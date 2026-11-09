"""Exercise the actual verifier with a deterministic ComfyUI transport fixture.

These tests validate report decisions, not GPU port correctness.
"""
import builtins
import io
import json
import pathlib
import re
import shutil
import tempfile
import types
import unittest
from unittest.mock import patch


SOURCE = pathlib.Path(__file__).resolve().parents[1] / 'portkit/kaggle/verify_run.py'


def verify(fault=None):
    clock = [0.0]
    events, samples, logs = [], [], []
    temp_root = SOURCE.parents[2] / 'portkit_runs'
    temp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
        out = pathlib.Path(tmp)
        log = out / 'source.log'

        def run(name, graph):
            clock[0] += 1
            if name.startswith('switch_off='):
                state = name.split('=')[1]
                names = [] if fault == 'empty_switch' else ['test_tensor_parallel']
                logs.append('[TPSwitch] exchange_off = %s modules %r' % (state, names))
                log.write_text('\n'.join(logs))
                return fault != 'switch_failed'
            if name == 'V5_release':
                return True
            if name == 'V1_cold':
                error = '0.5' if fault == 'block' else '0.002759'
                logs.append('[TP-CHECK] block0 rel_err=%s drift=0 finite=True' % error)
            logs.append('[FiniteCheck] non-finite values: %d' % (1 if fault == 'finite' else 0))
            # The VAE stream is deliberately slower: it must never stand in for sampler ticks.
            for value in (1, 2, 3):
                clock[0] += 22.1 if name == 'V3_warm_exchange_off' else 23.0
                node = 'decode' if fault == 'vae_only' else 'sample'
                events.append((clock[0], {'type': 'progress', 'data': {'node': node, 'max': 3, 'value': value}}))
                samples.append((clock[0], None, None, None, [[14.5 * 1024 if fault == 'peak' else 12 * 1024], [12 * 1024]]))
            log.write_text('\n'.join(logs))
            return not (fault == 'reload' and name == 'V6_after_release')

        ns = dict(KIT_GRAPH={'sample': {'class_type': 'KSampler', 'inputs': {'seed': 1}}},
                  EVENTS=events, SAMPLES=samples, time=types.SimpleNamespace(time=lambda: clock[0]),
                  run=run, say=lambda *a: None, LOG=log, OUT=out, re=re, json=json,
                  os=types.SimpleNamespace(path=types.SimpleNamespace(exists=lambda p: True)),
                  shutil=shutil, T0=0, STOP=types.SimpleNamespace(set=lambda: None),
                  proc=types.SimpleNamespace(terminate=lambda: None))
        probe = {'after': {'cuda:0': {'allocated_gib': 2.29},
                           'cuda:1': {'allocated_gib': 9 if fault == 'release' else 0.01}}}
        real_open = builtins.open

        def probe_open(path, *args, **kwargs):
            if str(path) == '/kaggle/working/h3_up/tpprobe.json':
                return io.StringIO(json.dumps(probe))
            return real_open(path, *args, **kwargs)

        with patch('builtins.open', probe_open):
            exec(compile(SOURCE.read_text(), str(SOURCE), 'exec'), ns)
        return (out / 'VERIFY_REPORT.md').read_text()


class VerifyDecisions(unittest.TestCase):
    def test_good_measurements_pass(self):
        self.assertTrue(verify().startswith('# VERIFY_REPORT: PASS'))

    def test_corrupted_measurements_fail(self):
        for fault in ('block', 'finite', 'peak', 'release', 'reload', 'empty_switch', 'switch_failed'):
            with self.subTest(fault=fault):
                self.assertTrue(verify(fault).startswith('# VERIFY_REPORT: FAIL'))

    def test_vae_progress_is_not_sampling_evidence(self):
        report = verify('vae_only')
        self.assertTrue(report.startswith('# VERIFY_REPORT: INCOMPLETE'))
        self.assertIn('**exchange share**: UNKNOWN', report)


if __name__ == '__main__':
    unittest.main()
