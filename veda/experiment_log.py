"""Durable, secret-free experiment logging; keep calls outside measured loops."""
import contextlib
import datetime
import json
from pathlib import Path
import time
import traceback

class ExperimentLog:
    def __init__(self,out):
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
    def event(self,event_type,**data):
        row=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),event=event_type,**data)
        with (self.out/'events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(row,default=str)+'\n');f.flush()
        print(json.dumps(row,default=str),flush=True)
    @contextlib.contextmanager
    def test(self,name,**data):
        self.event('START',test=name,**data);start=time.perf_counter()
        try:yield
        except Exception as exc:
            self.event('FAIL',test=name,error=repr(exc),traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-start)
            raise
        else:self.event('PASS',test=name,wall_seconds=time.perf_counter()-start)
    def tensor(self,name,t):
        self.event('TENSOR',name=name,shape=list(t.shape),dtype=str(t.dtype),device=str(t.device),stride=list(t.stride()),bytes=t.numel()*t.element_size())

