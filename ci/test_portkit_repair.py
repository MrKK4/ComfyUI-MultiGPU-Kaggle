"""Fresh repair tasks must retain the original model-specific task and deep evidence."""
import ast
import pathlib
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class RepairContext(unittest.TestCase):
    def test_cli_carries_original_task_and_deep_probe(self):
        runs = ROOT / 'portkit_runs'
        runs.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runs) as tmp:
            base = pathlib.Path(tmp)
            name = base.name
            original = 'LOCAL TEST FIXTURE: preserve the camera token and direct-block overrides.'
            (base / ('port_%s_prompt.txt' % name)).write_text(original)
            probe = base / 'probe'
            probe.mkdir()
            (probe / 'PROBE_REPORT.md').write_text('LOCAL TEST FIXTURE: probe report')
            (probe / 'probe.json').write_text('{"fixture_architecture": "camera-transformer"}')
            (probe / 'block_source.txt').write_text('def fixture_block(x): return x')
            code = base / 'broken.py'
            code.write_text('def supports(dm): return False\n')
            report = base / 'VERIFY_REPORT.md'
            report.write_text('LOCAL TEST FIXTURE: FAIL: supports rejected the target')
            subprocess.run([sys.executable, '-B', str(ROOT / 'portkit/tp_port.py'),
                            'repair', '--name', name, '--code', str(code),
                            '--verify-report', str(report), '--problem', 'wrong',
                            '--symptom', 'supports rejected the target'], check=True,
                           capture_output=True, text=True)
            task = base / 'attempt_2' / ('task_repair_%s_2.py' % name)
            tree = ast.parse(task.read_text(encoding='utf-8'))
            prompt = ast.literal_eval(tree.body[1].value)
            for text in (original, code.read_text(), report.read_text(),
                         (probe / 'probe.json').read_text(), (probe / 'block_source.txt').read_text()):
                self.assertIn(text, prompt)
            self.assertIn('everything you need is below', prompt)
            self.assertLess(len(prompt) / 3.4, 120000)


if __name__ == '__main__':
    unittest.main()
