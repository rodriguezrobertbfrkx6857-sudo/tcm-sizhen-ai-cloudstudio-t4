"""ROCm inventory stays read-only; only explicit smoke runs a tiny backward."""
import argparse,importlib.metadata,json,os,platform,re,shutil,subprocess,sys,time
from pathlib import Path
try:from .amd_common import finish_report,require_hip
except ImportError:from amd_common import finish_report,require_hip

def run_command(cmd):
    if not shutil.which(cmd[0]):return dict(status='NOT_INSTALLED')
    try:
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=15,check=False)
        return dict(status='OK' if p.returncode==0 else 'NONZERO',returncode=p.returncode,stdout=p.stdout[:12000],stderr=p.stderr[:1500])
    except (OSError,subprocess.TimeoutExpired):return dict(status='ERROR')

def inventory(workspace='/mnt/workspace',gpu_smoke=False,confirmed=False):
    report=dict(platform=platform.system(),python=sys.version.split()[0],gpu_smoke_requested=gpu_smoke,
                no_training_performed=True,no_network_or_install=True,gpu_model_claim='UNVERIFIED',
                persistent_mount_verified=False,persistent_across_restart='NOT_TESTED',
                env_presence_only={k:bool(os.getenv(k)) for k in ('HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES')})
    report['commands']={name:run_command(command) for name,command in {
        'rocminfo':['rocminfo'],'rocm_smi':['rocm-smi','--showproductname','--showmeminfo','vram'],
        'other_compute':['rocm-smi','--showpids'],'mount':['findmnt','-J','-T',str(workspace)]}.items()}
    report['gfx_targets']=sorted(set(re.findall(r'\bgfx[0-9a-f]+\b',report['commands']['rocminfo'].get('stdout',''))))
    report['devices_access']={name:dict(exists=Path(name).exists(),readable=os.access(name,os.R_OK),writable=os.access(name,os.W_OK)) for name in ['/dev/kfd','/dev/dri']}
    try:
        u=shutil.disk_usage(workspace);report['workspace_disk']=dict(exists=True,total_bytes=u.total,free_bytes=u.free,is_mount=Path(workspace).is_mount(),writable_permission=os.access(workspace,os.W_OK))
    except OSError:report['workspace_disk']=dict(exists=False)
    report['dependencies']={}
    for n in ['torch','torchvision','transformers','peft','accelerate','safetensors','ultralytics']:
        try:report['dependencies'][n]=importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:report['dependencies'][n]='NOT_INSTALLED'
    try:
        import torch
        report['torch']=dict(version=torch.__version__,hip=torch.version.hip,cuda_wheel_runtime=torch.version.cuda,gpu_available=torch.cuda.is_available())
        report['status']='READONLY_INVENTORY_NOT_ROCM' if not (torch.version.hip and torch.cuda.is_available()) else 'ROCM_VISIBLE_NUMERICS_PENDING'
        report['torch']['devices']=[]
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p=torch.cuda.get_device_properties(i);free,total=torch.cuda.mem_get_info(i)
                report['torch']['devices'].append(dict(index=i,name=p.name,gfx=getattr(p,'gcnArchName',None),total_bytes=total,free_bytes=free))
        if gpu_smoke:
            require_hip(torch,confirmed);torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
            torch.manual_seed(20261011);x=torch.randn((32,32),device='cuda',requires_grad=True)
            y=(x@x).square().mean();y.backward();torch.cuda.synchronize()
            if not torch.isfinite(y).item() or not torch.isfinite(x.grad).all().item():raise ValueError('NONFINITE_NUMERIC')
            report['gpu_smoke']=dict(status='PASS',loss=float(y.detach()),gradient_norm=float(x.grad.norm()),seconds=time.perf_counter()-t,peak_allocated_bytes=torch.cuda.max_memory_allocated())
            report['status']='ROCM_TINY_NUMERIC_PASS_NOT_MODEL_VALIDATION'
    except Exception as exc:
        report['status']='BLOCKED';report['error_type']=type(exc).__name__
        if gpu_smoke:report['gpu_smoke']=dict(status='FAIL',error_type=type(exc).__name__)
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',default='/mnt/workspace');p.add_argument('--gpu-smoke',action='store_true');p.add_argument('--confirm-no-other-compute',action='store_true');p.add_argument('--out');a=p.parse_args()
    r=inventory(a.workspace,a.gpu_smoke,a.confirm_no_other_compute);finish_report(r,a.out)
    return 2 if r['status']=='BLOCKED' or (a.gpu_smoke and r.get('gpu_smoke',{}).get('status')!='PASS') else 0
if __name__=='__main__':raise SystemExit(main())
