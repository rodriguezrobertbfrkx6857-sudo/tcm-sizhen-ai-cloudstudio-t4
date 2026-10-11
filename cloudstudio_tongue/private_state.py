"""Lightweight private artifact utilities without language-model dependencies."""
import json,os,time
from pathlib import Path

def safe_path(path):
    p=Path(path).resolve()
    if 'test2' in str(p).casefold():raise ValueError('SEALED_TEST2_ACCESS_FORBIDDEN')
    return p

def atomic(path,value):
    p=Path(path);tmp=p.with_name(p.name+'.partial')
    with tmp.open('w',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

class RunLock:
    def __init__(self,path):self.path=Path(path)
    def __enter__(self):
        fd=os.open(self.path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        with os.fdopen(fd,'w') as f:json.dump(dict(pid=os.getpid(),started=time.time()),f)
        return self
    def __exit__(self,*args):self.path.unlink()
