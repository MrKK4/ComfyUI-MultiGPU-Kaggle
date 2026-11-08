"""CPU regression checks of actual H3 methods, without importing ComfyUI/CUDA."""
import ast
import logging
import os
from pathlib import Path
import types
import unittest
import numpy as np

tree=ast.parse((Path(__file__).resolve().parents[1]/'h3_tensor_parallel.py').read_text())
ns={'torch':types.SimpleNamespace(float16=np.float16,empty=lambda shape,**kw:np.empty(shape,dtype=kw['dtype']),
        is_tensor=lambda x:isinstance(x,np.ndarray),cuda=types.SimpleNamespace(synchronize=lambda d:None,
        empty_cache=lambda:None,device=lambda d:__import__('contextlib').nullcontext())),
    'DEVICES':(0,1),'_SHARD_CACHE':{},'logger':logging.getLogger('test'),'QKV_ROWS':4096,
    'LORA_MODE':'auto','os':os,'PROFILE_FLAG':'__nonexistent_h3_test_flag__'}
ns['LORA_TARGETS']=("attn.qkv_proj","attn.out_proj","mlp.fc1","mlp.fc2")
exec(compile(ast.Module(body=[n for n in tree.body if (isinstance(n,ast.ClassDef) and n.name=='H3TensorParallel')
    or (isinstance(n,ast.FunctionDef) and n.name in ('_no_block_prefetch','_prepare_sampling'))],type_ignores=[]),'h3-methods','exec'),ns)
TP=ns['H3TensorParallel']

class Tensor(np.ndarray):
    device='cpu'
    def copy_(self,other):self[:]=other;return self

ns['torch'].empty=lambda shape,**kw:np.empty(shape,dtype=kw['dtype']).view(Tensor)

class MemoryTests(unittest.TestCase):
    def test_stack_switch_and_remove_adapters_without_allocating_gpu(self):
        class Adapter:
            def __init__(self,rank):self.weights=(types.SimpleNamespace(numel=lambda:768*rank),types.SimpleNamespace(numel=lambda:512*rank),None,None,None)
        ns['comfy']=types.SimpleNamespace(weight_adapter=types.SimpleNamespace(LoRAAdapter=Adapter))
        tp=TP.__new__(TP);tp.lora=None;tp.lora_sig=();tp.lora_entries={}
        a,b=Adapter(8),Adapter(16);key='diffusion_model.blocks.0.attn.qkv_proj.weight'
        tp.set_lora({key:[(0.8,a,1.0,None,None)]});first=tp.lora_bytes
        self.assertIsNone(tp.lora);self.assertEqual(first,(512*8+768*8//2)*2)
        tp.set_lora({key:[(0.8,a,1.0,None,None),(0.1,b,1.0,None,None)]})
        self.assertEqual(len(tp.lora_entries[0]['attn.qkv_proj']),2);self.assertEqual(tp.lora_bytes,first*3)
        tp.set_lora({});self.assertFalse(tp.lora_entries);self.assertEqual(tp.lora_bytes,0)
        tp.set_lora({key:[(0.8,a,1.0,None,None)]})
        with self.assertRaises(ValueError):tp.set_lora({key:[(0.8,a,0.5,None,None)]})

    def test_large_adapter_streams_when_activation_budget_would_not_fit(self):
        ns['comfy']=types.SimpleNamespace(model_management=types.SimpleNamespace(get_free_memory=lambda d:4<<30))
        tp=TP.__new__(TP);tp.lora_mode=None;tp.lora_bytes=1<<30
        tp._configure_lora(np.empty((40858,5376),dtype=np.uint8))
        self.assertEqual(tp.lora_mode,'stream')
        tp.lora_mode=None
        tp._configure_lora(np.empty((100,5376),dtype=np.uint8))
        self.assertEqual(tp.lora_mode,'resident')

    def test_forward_residual_state_cleared_even_on_error(self):
        opts={'_mmh3_tp':{'x1':object()},'unrelated':'keep'}
        def fail(*a,**kw):raise RuntimeError('model failure')
        with self.assertRaisesRegex(RuntimeError,'model failure'):
            ns['_no_block_prefetch'](fail,transformer_options=opts)
        self.assertNotIn('_mmh3_tp',opts);self.assertEqual(opts['unrelated'],'keep')

    def test_lora_validation_failure_still_releases_sampling_allocations(self):
        events=[]
        ns['comfy']=types.SimpleNamespace(model_management=types.SimpleNamespace(free_memory=lambda amount,d:events.append(('free',d))))
        def fail(patches):raise ValueError('unsupported LoRA')
        tp=types.SimpleNamespace(set_lora=fail,release=lambda:events.append(('release',)))
        executor=types.SimpleNamespace(class_obj=types.SimpleNamespace(model_patcher=types.SimpleNamespace(patches={})))
        with self.assertRaisesRegex(ValueError,'unsupported LoRA'):ns['_prepare_sampling'](tp,executor)
        self.assertEqual(events,[('free',0),('free',1),('release',)])

    def test_qkv_bounds_temporary_rows_with_lora(self):
        calls=[]
        def norm(x,w,eps,shift,scale,segs):
            calls.append(len(x));return np.asarray(x,dtype=np.float16)
        ns.update(_norm_mod=norm,_lin=lambda x,w:x@w,_lora=lambda x,p:(x@p[0])@p[1],
                  _chunk_segs=lambda seg,a,b:[(0,b-a,0)])
        tp=TP.__new__(TP);tp.eps1=1e-6
        x=np.ones((9001,4),dtype=np.float16).view(Tensor)
        w=np.ones((4,6),dtype=np.float16);pair=(np.ones((4,2),dtype=np.float16),np.ones((2,6),dtype=np.float16))
        out=tp._qkv(x,{'norm1':None,'qkv':w,'inner':2},None,None,[(0,len(x),0)],{'attn.qkv_proj':pair})
        np.testing.assert_array_equal(out,np.asarray(x)@w+(np.asarray(x)@pair[0])@pair[1])
        self.assertLessEqual(max(calls),4096,'QKV temporary still spans the whole video')

    def test_explicit_release_clears_cache_and_lora(self):
        tp=TP.__new__(TP);tp.path='model';tp.shards=object();tp.lora=object();tp.lora_sig=('old',)
        tp.lora_entries={1:'adapter'};tp.lora_mode=None;tp.xchg=types.SimpleNamespace(bufs={'old':object()})
        ns['_SHARD_CACHE']['model']=tp.shards
        tp.release(drop_shards=True)
        self.assertNotIn('model',ns['_SHARD_CACHE'],'Shard cache still owns supposedly released VRAM')
        self.assertIsNone(tp.shards);self.assertIsNone(tp.lora);self.assertEqual(tp.lora_sig,())
        self.assertFalse(tp.lora_entries);self.assertFalse(tp.xchg.bufs)

    def test_sampling_release_preserves_base_shards_but_frees_lora_gpu(self):
        tp=TP.__new__(TP);tp.path='warm';tp.shards=object();tp.lora=object();tp.lora_entries={1:'adapter'}
        tp.lora_mode='resident';tp.xchg=types.SimpleNamespace(bufs={})
        shards=tp.shards;tp.release()
        self.assertIs(tp.shards,shards);self.assertIsNone(tp.lora)
        self.assertEqual(tp.lora_entries,{1:'adapter'})

if __name__=='__main__':unittest.main()
