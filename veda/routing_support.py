"""Learned VEDA routing with experimental Kitchen SM75 sparse calculation."""
import ctypes as C
import hashlib
import importlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import threading
import time
import traceback
import torch
import comfy_kitchen as ck
from .experiment_log import ExperimentLog
from veda_comfy.core import bundle,engine,h3_layout,predictor,reference,selection,tiling
from veda_comfy.backends import base

OUT=Path('/kaggle/working/h3_pdmd_molmo');LOG=ExperimentLog(OUT);RESULTS=[]
def save(): (OUT/'results.json').write_text(json.dumps(RESULTS,indent=2))
def quantize(q,k,v):
    m=importlib.import_module('comfy_kitchen.sage_attention');old=m._select_cta_k;m._select_cta_k=lambda *a,**kw:128
    try:return ck.prequantize_int8_attention(q,k,v)
    finally:m._select_cta_k=old

class KitchenSparse(base.Backend):
    name='experimental-kitchen-sm75';display=name;dtypes=(torch.float16,);tolerance=.05
    def __init__(self,lib):self.lib=lib;self.seen=set()
    def attend(self,q,k,v,block_mask,layout):
        idx,count=selection.tile_index_list(block_mask & layout.kv_ok)
        qh,kh,vh=[x.permute(1,0,2).unsqueeze(0) for x in (q,k,v)]
        z=quantize(qh,kh,vh);n,h=q.shape[:2]
        out=torch.empty((1,h,n,128),device=q.device,dtype=torch.float16)
        signature=(str(q.device),n,h)
        if signature not in self.seen:
            LOG.event('SPARSE_LAUNCH_CONFIG',device=str(q.device),tokens=n,heads=h,tile=128,threads=[32,8],dynamic_shared_bytes=49152,index_shape=list(idx.shape),valid_shape=list(layout.valid_count.shape))
            self.seen.add(signature)
        err=self.lib.run_sparse(*[C.c_void_p(t.data_ptr()) for t in (z.q,z.k,z.v,out,z.q_scale,z.k_scale,z.v_scale)],
            n,h,z.v.shape[-1],*[C.c_void_p(t.data_ptr()) for t in (idx,count,layout.valid_count)],idx.shape[-1],C.c_void_p(torch.cuda.current_stream().cuda_stream))
        assert err==0,'Kernel launch error '+str(err)
        result=out[0].permute(1,0,2).contiguous()
        result.mul_(layout.slot_valid[:,None,None])
        return result

class RankEngine(engine.VedaEngine):
    """Original VEDA routing semantics with explicit global-to-local head IDs.

    Derived from VEDA engine.py at67e4eb5541b648555563144401ebaba832bd73ec,
    MIT license in the pinned remote checkout. Original source remains unchanged.
    """
    def __init__(self,b,backend,device,global_heads,generated=None,reference_budget=None):
        super().__init__(b,generated or b.generated,reference_budget or b.reference,backend,device)
        self.global_heads=list(global_heads);self.lookup={g:i for i,g in enumerate(self.global_heads)}
        self.capture=False;self.routes={};self._host_layer=None;self._host_pair=None;self._rank_groups={}
    def rank_groups(self,plan,layer):
        key=(id(plan),layer)
        if key not in self._rank_groups:
            groups=[]
            for group in plan.head_groups(layer,self.device):
                local=[self.lookup[g] for g in group.heads.tolist() if g in self.lookup]
                if local:groups.append((group.shape,torch.tensor(local,device=self.device,dtype=torch.long)))
            self._rank_groups[key]=groups
        return self._rank_groups[key]
    def _rank_weights(self,layer):
        if self._host_layer!=layer:
            ids=torch.tensor(self.global_heads,dtype=torch.long)
            self._host_pair=tuple(x[layer].index_select(0,ids).contiguous().pin_memory() for x in (self.bundle.proj_q,self.bundle.proj_k))
            self._host_layer=layer
        return tuple(x.to(self.device,non_blocking=True) for x in self._host_pair)
    @torch.no_grad()
    def attention(self,q,k,v,layer,spec,plan,head_chunks=1):
        timer=self.timer;s,h,d=q.shape;out=q.new_empty(s+1,h,d)
        with timer('predictor_weights'):pq,pk=self._rank_weights(layer)
        chunk_bytes=self._chunk_bytes();chunks=0
        for shape,local in self.rank_groups(plan,layer):
            # Plans/projections use original head IDs. Only QKV gather indices
            # are rebased to the rank-local tensor.
            layout=self._tile_layout(spec,shape)
            blocks=selection.column_blocks(layout,self.generated,self.reference)
            per_head=layout.num_slots*d*q.element_size()
            step=max(1,min(chunk_bytes//per_head,max(1,out.numel()*out.element_size()//(per_head*6))))
            if head_chunks>1:step=min(step,max(1,(h+head_chunks-1)//head_chunks))
            for ids in local.split(step):
                chunks+=1
                with timer('gather_qk'):
                    qt=tiling.gather_tiles(q,layout,ids);kt=tiling.gather_tiles(k,layout,ids)
                with timer('score'):
                    scores=predictor.tile_logits(predictor.pool_video_tiles(qt,layout),predictor.pool_video_tiles(kt,layout),pq.index_select(0,ids),pk.index_select(0,ids))
                with timer('selection'):
                    index,keep=selection.select(scores,layout,blocks);del scores
                    mask=selection.block_mask(index,keep,layout);self.stats.add(keep,layout)
                if self.capture:
                    for i,j in enumerate(ids.tolist()):self.routes[layer,self.global_heads[j]]=mask[i].detach().cpu()
                with timer('gather_v'):vt=tiling.gather_tiles(v,layout,ids)
                with timer('quantize_index_sparse'):ot=self.backend.attend(qt,kt,vt,mask,layout)
                with timer('scatter'):tiling.scatter_tiles_(out,ot,layout,ids)
                del qt,kt,vt,ot,mask
        self.chunking['chunks_per_layer']=chunks
        return out[:s]

