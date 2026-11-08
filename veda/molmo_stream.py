"""Stream original Molmo tensors one execution unit at a time, without casting.

CPU tensors are retained by reference, not copied back from GPU. Hooks restore
the original objects even when a forward raises. Only text-conditioning is
supported here; no vision/generation-path claim.
"""
from contextlib import contextmanager
import time
import torch


def units(hf):
    text=hf.model.transformer
    result=[('embedding',text.wte)]
    if hasattr(text,'rotary_emb'):result.append(('rope',text.rotary_emb))
    else:result.extend(('rope_'+k,m) for k,m in text.rotary_embs.items())
    result.extend((f'block_{i}',block) for i,block in enumerate(text.blocks))
    result.extend([('final_norm',text.ln_f),('lm_head',hf.lm_head)])
    return result


@contextmanager
def streamed(hf,device,log):
    handles=[];active={};stats=dict(transferred_bytes=0,units=0,transfer_wall_seconds=0.0)

    def restore(module):
        saved=active.pop(id(module),[])
        for owner,kind,key,original in reversed(saved):getattr(owner,kind)[key]=original

    def before(label):
        def hook(module,args):
            if id(module) in active:raise RuntimeError('Nested streaming unit '+label)
            saved=[];active[id(module)]=saved;start=time.perf_counter();nbytes=0
            try:
                for owner in module.modules():
                    for kind in ('_parameters','_buffers'):
                        for key,value in list(getattr(owner,kind).items()):
                            if value is None:continue
                            assert value.device.type=='cpu',f'{label}/{key} must start on CPU'
                            copied=value.to(device=device,non_blocking=False)
                            if kind=='_parameters':copied=torch.nn.Parameter(copied,requires_grad=value.requires_grad)
                            saved.append((owner,kind,key,value));getattr(owner,kind)[key]=copied
                            nbytes+=value.numel()*value.element_size()
            except BaseException:
                restore(module);raise
            stats['units']+=1;stats['transferred_bytes']+=nbytes
            stats['transfer_wall_seconds']+=time.perf_counter()-start
            log.event('MOLMO_STREAM_UNIT',unit=label,bytes=nbytes)
        return hook

    def after(module,args,output):restore(module)

    try:
        for label,module in units(hf):
            handles.append(module.register_forward_pre_hook(before(label)))
            handles.append(module.register_forward_hook(after,always_call=True))
        yield stats
    finally:
        for handle in handles:handle.remove()
        # Post hooks with always_call handle exceptions inside normal forward;
        # this also covers a partially installed hook or a pre-hook failure.
        for _,module in units(hf):restore(module)
        assert not active,'Streaming left unrecovered module tensors'


def hidden(runtime,ids,tap,log,device):
    with runtime.lock:
        runtime.load()
        hf=runtime.model.transformer
        tokens=torch.tensor([ids],device=device)
        with torch.inference_mode(),streamed(hf,device,log) as stats:
            output=runtime.model(input_ids=tokens,attention_mask=torch.ones_like(tokens),
                output_hidden_states=True,use_cache=False,logits_to_keep=1)
            result=output.hidden_states[int(tap)][0]
            assert result.shape[0]==len(ids),'Streaming changed token alignment'
            result=result.float().to('cpu',dtype=torch.float16)
        log.event('MOLMO_STREAM_SUMMARY',**stats)
        assert all(t.device.type=='cpu' for t in list(hf.parameters())+list(hf.buffers())),'Molmo weights survived on GPU'
        return result
