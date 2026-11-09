"""Short real-GPU Idol port trial, bypassing image encoders and decoders.
Reports only checks actually measured; this is not end-to-end workflow validation.
"""
import argparse, gc, importlib, json, logging, os, pathlib, sys, time, types
import torch

ap=argparse.ArgumentParser()
ap.add_argument('--comfy',required=True)
ap.add_argument('--fork',required=True)
ap.add_argument('--module',required=True)
ap.add_argument('--model',required=True)
ap.add_argument('--out',required=True)
a=ap.parse_args()
out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,a.comfy)
# ComfyUI must not parse this test runner's command-line flags.
sys.argv=[sys.argv[0]]
logging.basicConfig(level=logging.INFO)
import comfy.sd, comfy.model_management
pkg=types.ModuleType('port_trial');pkg.__path__=[a.fork];sys.modules[pkg.__name__]=pkg
source=importlib.import_module('port_trial.tp_source')
auto=importlib.import_module('port_trial.tp_auto')
spec=importlib.util.spec_from_file_location('port_trial.idol_tensor_parallel',a.module)
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
assert torch.cuda.device_count()==2
patcher=comfy.sd.load_diffusion_model(a.model,model_options={'dtype':torch.float16},disable_dynamic=True)
comfy.model_management.load_model_gpu(patcher)
dm=patcher.model.diffusion_model
checks=[];measurements={}
def check(name,ok,detail):
 checks.append({'name':name,'status':'PASS' if ok else 'FAIL','detail':detail})
def sync():
 for i in (0,1):torch.cuda.synchronize(i)
def inputs(seed):
 torch.manual_seed(seed)
 return [torch.randn(1,8192,16,device='cuda:0',dtype=torch.float16),torch.randn(1,1,5,device='cuda:0',dtype=torch.float16)],torch.tensor([0.5],device='cuda:0'),torch.randn(1,64,1280,device='cuda:0',dtype=torch.float16)
def forward(seed):
 x,t,c=inputs(seed)
 sync();t0=time.perf_counter()
 with torch.inference_mode():y=dm(x,t,context=c,transformer_options={})
 sync();secs=time.perf_counter()-t0
 return [v.detach().float().cpu() for v in y],secs
refs={}
for seed in (1,2,3):
 y,secs=forward(seed);refs[seed]=y
 measurements.setdefault('single_gpu_seconds',[]).append(secs)
check('supports target',mod.supports(dm),type(dm).__name__)
check('rejects other models',not mod.supports(torch.nn.Linear(4,4)),None)
# Use the registry path, not a parallel private attach implementation.
auto.MODULES=(('Idol',mod,'idol_tp'),)+auto.MODULES
try:
 tp=auto.attach(patcher)
 check('automatic attach',tp is not None,None)
 for i in (0,1):torch.cuda.reset_peak_memory_stats(i)
 for seed in (1,2,3):
  y,secs=forward(seed)
  ref=torch.cat([v.flatten() for v in refs[seed]])
  got=torch.cat([v.flatten() for v in y])
  finite=bool(torch.isfinite(got).all())
  cosine=float(torch.nn.functional.cosine_similarity(ref,got,dim=0)) if finite else float('nan')
  rel=float(torch.linalg.vector_norm(got-ref)/torch.linalg.vector_norm(ref).clamp_min(1e-10))
  check('full forward seed %d'%seed,finite and cosine>=.99,dict(finite=finite,cosine=cosine,rel_err=rel))
  measurements.setdefault('tp_seconds',[]).append(secs)
 measurements['peak_allocated_gib']=[torch.cuda.max_memory_allocated(i)/2**30 for i in (0,1)]
 check('memory margin',all(torch.cuda.get_device_properties(i).total_memory/2**30-measurements['peak_allocated_gib'][i]>=1 for i in (0,1)),measurements['peak_allocated_gib'])
 if hasattr(mod,'XCHG_OFF'):
  original=mod.XCHG_OFF
  mod.XCHG_OFF=True
  broken,secs=forward(1)
  mod.XCHG_OFF=original
  ref=torch.cat([v.flatten() for v in refs[1]])
  got=torch.cat([v.flatten() for v in broken])
  cosine=float(torch.nn.functional.cosine_similarity(ref,got,dim=0))
  negative_pass=bool(torch.isfinite(got).all()) and cosine>=.99
  measurements['deliberate_exchange_off']={'cosine':cosine,'verifier_verdict':'PASS' if negative_pass else 'FAIL','seconds':secs}
  check('deliberately broken port is rejected',not negative_pass,measurements['deliberate_exchange_off'])
 else:check('exchange switch exists',False,None)
 tp.release(drop_shards=True)
 gc.collect()
 for i in (0,1):torch.cuda.empty_cache()
 measurements['after_release_allocated_gib']=[torch.cuda.memory_allocated(i)/2**30 for i in (0,1)]
 check('release under 3 GiB',all(v<3 for v in measurements['after_release_allocated_gib']),measurements['after_release_allocated_gib'])
 auto.attach(patcher)
 y,secs=forward(1)
 got=torch.cat([v.flatten() for v in y]);ref=torch.cat([v.flatten() for v in refs[1]])
 check('reload reproduces output',bool(torch.isfinite(got).all()) and float(torch.nn.functional.cosine_similarity(ref,got,dim=0))>=.99,secs)
except Exception as e:
 import traceback
 check('runtime',False,traceback.format_exc())
verdict='PASS' if all(c['status']=='PASS' for c in checks) else 'FAIL'
measurements['speed_note']='Compare warm timings only. A slower port requires an explicit fallback decision; no speed acceptance is inferred.'
report={'verdict':verdict,'checks':checks,'measurements':measurements,'scope':'Direct denoiser outputs, synthetic inputs; image encoders, decoder, sampler hooks and production workflows are not exercised.'}
(out/'verify.json').write_text(json.dumps(report,indent=2))
md='# VERIFY_REPORT: '+verdict+'\n\n'+report['scope']+'\n\n'+''.join('- **'+c['name']+'**: '+c['status']+': '+str(c['detail'])+'\n' for c in checks)+'\n```json\n'+json.dumps(measurements,indent=2)+'\n```\n'
(out/'VERIFY_REPORT.md').write_text(md)
print(md,flush=True)
