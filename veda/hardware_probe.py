"""Hardware observations and small throughput ceilings, in the ComfyUI process."""
import gc
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import threading
import time
import torch

OUT=Path('/kaggle/working/h3_attention_stages')

def timed(fn,device,repeats=5):
    for _ in range(3):fn()
    torch.cuda.synchronize(device);times=[]
    for _ in range(repeats):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        a.record();result=fn();b.record();b.synchronize();times.append(a.elapsed_time(b));del result
    return statistics.median(times)

def telemetry():
    fields=['timestamp','index','pstate','utilization.gpu','utilization.memory','memory.used','memory.total',
            'clocks.sm','clocks.mem','clocks.max.sm','clocks.max.mem','power.draw','power.limit','temperature.gpu',
            'pcie.link.gen.current','pcie.link.width.current',
            'clocks_event_reasons.sw_power_cap','clocks_event_reasons.hw_thermal_slowdown',
            'clocks_event_reasons.hw_power_brake_slowdown']
    try:
        helptext=subprocess.check_output(['nvidia-smi','--help-query-gpu'],text=True)
        fields=[f for f in fields if f in helptext]
        (OUT/'hardware_telemetry_fields.json').write_text(json.dumps(fields,indent=2))
        while True:
            t=time.time()
            r=subprocess.run(['nvidia-smi','--query-gpu='+','.join(fields),'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8)
            row=dict(time=t,rows=r.stdout.strip().splitlines(),error=r.stderr.strip() if r.returncode else None)
            with (OUT/'hardware_telemetry.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            time.sleep(1)
    except Exception as exc:
        (OUT/'hardware_telemetry_error.txt').write_text(str(exc))

def run():
    OUT.mkdir(exist_ok=True)
    threading.Thread(target=telemetry,daemon=True).start()
    result=dict(devices=[],p2p=torch.cuda.can_device_access_peer(0,1),ncu_path=shutil.which('ncu'),
                notes=['Microbenchmark ceilings are not a single percentage for an entire mixed H3 workflow.',
                       'GPU-copy GB/s counts both bytes read and written; host-transfer GB/s counts payload once.',
                       'INT8 _int_mm reports matrix multiplication work; FP16 reports GEMM work. H3 also does other operations.'])
    for name in ('memory.max','memory.current'):
        p=Path('/sys/fs/cgroup')/name
        if p.exists():result['cgroup_'+name]=p.read_text().strip()
    with torch.inference_mode():
        n=256*1024*1024
        host=torch.empty(n,dtype=torch.uint8,pin_memory=True);host.fill_(17)
        for d in (0,1):
            dev=torch.device('cuda',d)
            with torch.cuda.device(dev):
                prop=torch.cuda.get_device_properties(dev)
                row=dict(device=d,name=prop.name,sm_count=prop.multi_processor_count,total_memory=prop.total_memory)
                x=torch.empty(n,dtype=torch.uint8,device=dev);y=torch.empty_like(x)
                for label,fn,traffic in [('h2d',lambda:x.copy_(host,non_blocking=True),n),
                                         ('d2h',lambda:host.copy_(x,non_blocking=True),n),
                                         ('vram_copy',lambda:y.copy_(x),2*n)]:
                    ms=timed(fn,dev);row[label]=dict(ms=ms,GB_per_second=traffic/ms/1e6)
                del x,y
                a=torch.randn(4096,4096,device=dev,dtype=torch.float16);b=torch.randn_like(a)
                ms=timed(lambda:a@b,dev);row['fp16_gemm']=dict(ms=ms,TFLOPS=2*4096**3/ms/1e9)
                del a,b
                try:
                    a=torch.randint(-127,128,(4096,4096),device=dev,dtype=torch.int8);b=torch.randint_like(a,-127,128)
                    ms=timed(lambda:torch._int_mm(a,b),dev);row['int8_gemm']=dict(ms=ms,TOPS=2*4096**3/ms/1e9)
                except Exception as exc:row['int8_gemm']=dict(error=str(exc))
                finally:del a,b
                result['devices'].append(row);gc.collect();torch.cuda.empty_cache()
        del host
    (OUT/'hardware_probe.json').write_text(json.dumps(result,indent=2))
    print('[H3 HARDWARE PROBE]',json.dumps(result),flush=True)
    return result
