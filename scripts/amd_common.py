"""Shared immutable reports and explicit HIP device checks; no installs/network."""
import datetime, hashlib, json, os, subprocess, uuid
from pathlib import Path

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def finish_report(value, out=None):
    entry=Path(__import__('sys').argv[0])
    value=dict(value, timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               run_id='LOCAL_OR_CLOUD_PROBE_'+uuid.uuid4().hex,
               optimizer_updates=0, model_downloads=False, cloud_training=False,
               entrypoint_SHA256=sha256(entry) if entry.is_file() else None,
               report_writer_SHA256=sha256(__file__))
    try:
        value['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True,stderr=subprocess.DEVNULL).strip()
        value['working_tree_dirty']=bool(subprocess.check_output(['git','status','--porcelain'],cwd=Path(__file__).resolve().parents[1],text=True))
    except (OSError,subprocess.CalledProcessError):value['git_commit']=None
    data=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if out:
        p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
        # Atomic, exclusive publication; never replace a previous observation.
        tmp=p.parent/('.'+p.name+'.'+uuid.uuid4().hex+'.partial')
        try:
            with tmp.open('xb') as f:f.write(data.encode('utf8'));f.flush();os.fsync(f.fileno())
            os.link(tmp,p)
        finally:
            if tmp.exists():tmp.unlink()
        readback=p.read_bytes()
        assert readback==data.encode('utf8')
        with Path(str(p)+'.sha256').open('x',encoding='ascii',newline='\n') as f:f.write(hashlib.sha256(readback).hexdigest()+'\n')
    print(data)
    return value

def require_hip(torch, confirmed=False, min_free_gib=1):
    if not confirmed:raise ValueError('EXPLICIT_GPU_SMOKE_CONFIRMATION_REQUIRED')
    if not torch.version.hip or not torch.cuda.is_available():raise ValueError('BLOCKED_NON_ROCM_OR_NO_GPU')
    kfd=Path('/dev/kfd')
    if not kfd.exists() or not os.access(kfd,os.R_OK|os.W_OK):raise ValueError('KFD_READ_WRITE_PERMISSION_REQUIRED')
    free,total=torch.cuda.mem_get_info(0)
    if free<min_free_gib*1024**3:raise ValueError('INSUFFICIENT_REAL_FREE_VRAM')
    return dict(name=torch.cuda.get_device_name(0),free_bytes=free,total_bytes=total,
                other_compute='USER_CONFIRMED_NO_OTHER_COMPUTE_NOT_AUTOMATICALLY_PROVEN')
