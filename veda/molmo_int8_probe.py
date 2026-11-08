"""Reproduce Molmo's rejected skinny INT8 GEMM; validate aligned fix on T4s."""
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from experiment_log import ExperimentLog

out=Path(sys.argv[1]);root=Path(sys.argv[2]);log=ExperimentLog(out)
manifest=json.loads((root/'convrot_config.json').read_text())
shapes=sorted({(s['in_features'],s['out_features']) for s in manifest['modules'].values()})
rows=[]
for device in (0,1):
    with torch.cuda.device(device),torch.inference_mode():
        # Exact failure geometry: 40 tokens, 4096 input, 6144 output.
        q=torch.randint(-3,4,(40,4096),dtype=torch.int8,device='cuda')
        w=torch.randint(-3,4,(6144,4096),dtype=torch.int8,device='cuda')
        for name,align,column_major in [('original',8,False),('align32',32,False),('transpose_view',8,True)]:
            try:
                a=F.pad(q,(0,0,0,((q.shape[0]+align-1)//align)*align-q.shape[0]))
                b=w.t() if column_major else w.t().contiguous()
                y=torch._int_mm(a,b);torch.cuda.synchronize()
                ref=q[:8].float()@w[:8].float().t()
                assert torch.equal(y[:8,:8].float(),ref)
                row=dict(device=device,variant=name,status='PASS',input_shape=list(a.shape),weight_stride=list(b.stride()))
            except RuntimeError as e:
                row=dict(device=device,variant=name,status='REJECTED',error=str(e))
            rows.append(row);log.event('MOLMO_INT8_VARIANT',**row)
        del q,w,a,b
        for k,n in shapes:
            with log.test('molmo_int8_aligned_shape',device=device,in_features=k,out_features=n):
                w=torch.randint(-3,4,(n,k),dtype=torch.int8,device='cuda')
                for tokens in (1,40,64):
                    q=torch.randint(-3,4,(tokens,k),dtype=torch.int8,device='cuda')
                    a=F.pad(q,(0,0,0,max(32,((tokens+31)//32)*32)-tokens))
                    b=w.t().contiguous()
                    y=torch._int_mm(a,b)[:tokens];torch.cuda.synchronize()
                    ref=q[:8].float()@w[:8].float().t()
                    assert torch.equal(y[:8,:8].float(),ref),(device,k,n,tokens)
                    log.event('MOLMO_INT8_SHAPE_PASS',device=device,in_features=k,out_features=n,tokens=tokens)
                    del q,a,b,y,ref
                del w
        torch.cuda.empty_cache()
(out/'molmo_int8_preflight.json').write_text(json.dumps(dict(status='PASS',variants=rows,validated_shapes=shapes,devices=[0,1]),indent=2))
