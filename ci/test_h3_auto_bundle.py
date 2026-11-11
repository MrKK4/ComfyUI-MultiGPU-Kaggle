"""Exercise notebook routing and managed-memory interception without ComfyUI."""

import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]


class AutoBundleTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(lambda: [sys.modules.pop(name, None) for name in self.modules])
        self.modules = []
        self.calls = []

        def module(name):
            item = types.ModuleType(name)
            sys.modules[name] = item
            self.modules.append(name)
            return item

        package = module('h3_auto_fixture')
        package.__path__ = [str(ROOT)]
        source = module(package.__name__ + '.h3_bundle_source')
        source.H3BundleSource = type('H3BundleSource', (), {
            '__init__': lambda self, *args: setattr(self, 'args', args)})
        loader = module(package.__name__ + '.h3_bundle_loader')
        loader.load_h3_bundle = lambda *args: ('bundle', args)
        parallel = module(package.__name__ + '.h3_tensor_parallel')
        parallel._INSTANCES = {}
        parallel._BUNDLE_LEASES = {}
        parallel.evict_bundle_tp = lambda tp, targets: self.calls.append(('evict', targets)) or (0, 2)
        parallel.UNETLoaderH3TensorParallel = type('UNETLoaderH3TensorParallel', (), {
            'load': lambda self, name: ('ordinary TP', name)})
        package.h3_tensor_parallel = parallel

        paths = module('folder_paths')
        paths.get_filename_list = lambda category: ['plain.safetensors', 'Minimax-h3_Singularity_ref2va_v2_pruned_int8.safetensors'] if category == 'diffusion_models' else []
        nodes = module('nodes')
        nodes.UNETLoader = type('UNETLoader', (), {
            'load_unet': lambda self, name, dtype: ('ordinary', name, dtype)})
        comfy = module('comfy')
        comfy.__path__ = []
        mm = module('comfy.model_management')
        mm.free_memory = lambda needed, device, *args, **kwargs: self.calls.append(('free', needed, device)) or ['unloaded']
        comfy.model_management = mm

        name = package.__name__ + '.h3_auto_bundle'
        spec = importlib.util.spec_from_file_location(name, ROOT / 'h3_auto_bundle.py')
        self.auto = importlib.util.module_from_spec(spec)
        sys.modules[name] = self.auto
        self.modules.append(name)
        spec.loader.exec_module(self.auto)
        self.paths, self.nodes, self.mm, self.parallel = paths, nodes, mm, parallel

    def test_h3_loader_routing_and_other_models(self):
        env = {'MMH3_TP_BUNDLE_AUTO': '1', 'MMH3_TP_BUNDLE_DIR': 'bundle',
               'MMH3_TP_PIECES_DIR': 'pieces', 'MMH3_TP_BUNDLE_MANIFEST_SHA256': 'a' * 64,
               'MMH3_TP_BUNDLE_NAME': 'Olivia_H3_split_bundle.safetensors'}
        with patch.dict(os.environ, env), patch.object(torch.cuda, 'device_count', return_value=2):
            self.assertTrue(self.auto.install())
            router = self.nodes.UNETLoader.load_unet.__mmh3_bundle_router__
            self.assertTrue(router.is_routed('Olivia_H3_split_bundle.safetensors'))
            self.assertFalse(router.is_routed('plain.safetensors'))
            self.assertFalse(router.is_routed('Minimax-h3_Singularity_ref2va_v2_pruned_int8.safetensors'))
            self.assertEqual(router.bundle_key('Olivia_H3_split_bundle.safetensors').args,
                             ('bundle', 'pieces', 'a' * 64))
            self.assertIn('Olivia_H3_split_bundle.safetensors', self.paths.get_filename_list('diffusion_models'))
            self.assertNotIn('PRISM_DMAD_PDMD_int8.safetensors', self.paths.get_filename_list('diffusion_models'))
            self.assertEqual(self.nodes.UNETLoader().load_unet('Olivia_H3_split_bundle.safetensors', 'default')[0][0], 'bundle')
            self.assertEqual(self.parallel.UNETLoaderH3TensorParallel().load('Olivia_H3_split_bundle.safetensors')[0][0], 'bundle')
            self.assertEqual(self.nodes.UNETLoader().load_unet('plain.safetensors', 'default')[0], 'ordinary')
            self.assertEqual(self.nodes.UNETLoader().load_unet('Minimax-h3_Singularity_ref2va_v2_pruned_int8.safetensors', 'default')[0], 'ordinary')
            self.assertEqual(self.parallel.UNETLoaderH3TensorParallel().load('Minimax-h3_Singularity_ref2va_v2_pruned_int8.safetensors')[0], 'ordinary TP')
            self.assertFalse(self.nodes.UNETLoader.VALIDATE_INPUTS('Unknown_Minimax_H3.safetensors') is True)
            self.assertTrue(self.nodes.UNETLoader.VALIDATE_INPUTS('Minimax-h3_Singularity_ref2va_v2_pruned_int8.safetensors'))
            with self.assertRaisesRegex(ValueError, 'default'):
                self.nodes.UNETLoader().load_unet('Olivia_H3_split_bundle.safetensors', 'fp8_e5m2')

    def test_managed_request_evicts_only_requested_gpu_and_skips_sample_lease(self):
        env = {'MMH3_TP_BUNDLE_AUTO': '1', 'MMH3_TP_BUNDLE_DIR': 'bundle',
               'MMH3_TP_PIECES_DIR': 'pieces', 'MMH3_TP_BUNDLE_MANIFEST_SHA256': 'a' * 64,
               'MMH3_TP_BUNDLE_NAME': 'Olivia_H3_split_bundle.safetensors'}
        with patch.dict(os.environ, env), patch.object(torch.cuda, 'device_count', return_value=2):
            self.auto.install()
            source = self.auto.H3BundleSource()
            tp = types.SimpleNamespace(path=source, shards=[[object()], [object()]])
            self.parallel._INSTANCES['bundle'] = tp
            self.parallel._BUNDLE_LEASES[source] = 0
            with patch.object(torch.cuda, 'mem_get_info', return_value=(1 << 30, 15 << 30)):
                self.assertEqual(self.mm.free_memory(3 << 30, torch.device('cuda:1')), ['unloaded'])
                self.assertEqual(self.calls[-1][0], 'evict')
                self.assertEqual(self.calls[-1][1][0], 0.0)
                self.assertGreater(self.calls[-1][1][1], 3.0)
                before = len(self.calls)
                self.parallel._BUNDLE_LEASES[source] = 1
                self.mm.free_memory(3 << 30, torch.device('cuda:1'))
                self.assertEqual(len(self.calls), before + 1)  # only original free_memory
                self.parallel._BUNDLE_LEASES[source] = 0
                self.mm.free_memory(1e30, torch.device('cuda:1'))
                self.assertEqual(self.calls[-1][0], 'free')


if __name__ == '__main__':
    unittest.main()
