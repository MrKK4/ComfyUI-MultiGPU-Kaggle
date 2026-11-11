"""CPU checks for standard LoRA mapping into source-free H3 TP projections."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import torch


ROOT = Path(__file__).resolve().parents[1]


class _CorePatcher:
    def __init__(self, model=None, **kwargs):
        self.model = model
        self.patches = {}
        self.attachments = {}
        self.patches_uuid = object()

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        accepted = []
        for key, value in patches.items():
            if key in self.model.state_dict():
                self.patches.setdefault(key, []).append((strength_patch, value, strength_model, None, None))
                accepted.append(key)
        return accepted

    def clone(self):
        other = self.__class__(self.model)
        other.patches = {key: values[:] for key, values in self.patches.items()}
        other.attachments = {key: value.on_model_patcher_clone() for key, value in self.attachments.items()}
        return other

    def get_attachment(self, key):
        return self.attachments.get(key)

    def set_attachments(self, key, value):
        self.attachments[key] = value


class _LoRAAdapter:
    def __init__(self):
        self.weights = (torch.empty(4, 2), torch.empty(2, 4), None, None, None, None)


class BundleLoraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        comfy = types.ModuleType('comfy')
        comfy.lora = types.ModuleType('comfy.lora')
        comfy.model_patcher = types.ModuleType('comfy.model_patcher')
        comfy.weight_adapter = types.ModuleType('comfy.weight_adapter')
        comfy.lora.model_lora_keys_unet = lambda model, key_map={}: dict(key_map)
        comfy.model_patcher.CoreModelPatcher = _CorePatcher
        comfy.weight_adapter.LoRAAdapter = _LoRAAdapter
        with patch.dict(sys.modules, {'comfy': comfy, 'comfy.lora': comfy.lora,
                                      'comfy.model_patcher': comfy.model_patcher,
                                      'comfy.weight_adapter': comfy.weight_adapter}):
            spec = importlib.util.spec_from_file_location('_bundle_lora_test', ROOT / 'h3_bundle_lora.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        cls.bridge = module
        cls.comfy = comfy

    def setUp(self):
        self.model = types.SimpleNamespace(_mmh3_bundle_virtual_lora=True,
            state_dict=lambda: {'diffusion_model.final_layer.video_out.weight': object()})
        self.bridge.install_lora_key_bridge()

    def test_normal_lora_mapping_accepts_all_200_tp_keys(self):
        mapping = self.comfy.lora.model_lora_keys_unet(self.model, {})
        aliases = [key for key in mapping if key.startswith('lora_unet_blocks_')]
        self.assertEqual(len(aliases), 200)
        self.assertEqual(mapping['lora_unet_blocks_24_attn_out_proj'],
                         'diffusion_model.blocks.24.attn.out_proj.weight')
        ordinary = types.SimpleNamespace(state_dict=lambda: {})
        self.assertFalse(self.comfy.lora.model_lora_keys_unet(ordinary, {}))
        self.assertFalse(self.comfy.lora.model_lora_keys_unet(ordinary))
        self.assertEqual(mapping['transformer.blocks.24.attn.out_proj'],
                         'diffusion_model.blocks.24.attn.out_proj.weight')

    def test_clone_stack_and_remove_keep_tp_weights_out_of_core_patches(self):
        key = 'diffusion_model.blocks.24.attn.out_proj.weight'
        ordinary = 'diffusion_model.final_layer.video_out.weight'
        base = self.bridge.H3BundleModelPatcher(self.model)
        a, b = _LoRAAdapter(), _LoRAAdapter()
        first = base.clone()
        self.assertCountEqual(first.add_patches({key: a, ordinary: a}, 0.8), [key, ordinary])
        second = first.clone()
        self.assertEqual(second.add_patches({key: b}, 0.4), [key])
        self.assertNotIn(key, second.patches)
        self.assertEqual(len(self.bridge.sampling_patches(second)[key]), 2)
        self.assertEqual(len(self.bridge.sampling_patches(first)[key]), 1)
        self.assertNotIn(key, self.bridge.sampling_patches(base))
        self.assertIn(ordinary, second.patches)

    def test_invalid_tp_patch_raises_before_any_mutation(self):
        key = 'diffusion_model.blocks.24.attn.out_proj.weight'
        ordinary = 'diffusion_model.final_layer.video_out.weight'
        patcher = self.bridge.H3BundleModelPatcher(self.model)
        with self.assertRaises(self.bridge.H3BundleLoraError):
            patcher.add_patches({ordinary: _LoRAAdapter(), key: ('diff', object())})
        self.assertFalse(patcher.patches)
        self.assertFalse(patcher.attachments)
        with self.assertRaises(self.bridge.H3BundleLoraError):
            patcher.add_patches({key[:-len('weight')] + 'bias': _LoRAAdapter()})
        with self.assertRaises(self.bridge.H3BundleLoraError):
            patcher.add_patches({(key, (0, 1)): _LoRAAdapter()})
        dora = _LoRAAdapter()
        dora.weights = (*dora.weights[:4], torch.ones(4), None)
        with self.assertRaises(self.bridge.H3BundleLoraError):
            patcher.add_patches({key: dora})

    def test_sampling_map_is_independent(self):
        key = 'diffusion_model.blocks.24.attn.out_proj.weight'
        patcher = self.bridge.H3BundleModelPatcher(self.model)
        patcher.add_patches({key: _LoRAAdapter()})
        sampled = self.bridge.sampling_patches(patcher)
        sampled[key].clear()
        self.assertEqual(len(self.bridge.sampling_patches(patcher)[key]), 1)


if __name__ == '__main__':
    unittest.main()
