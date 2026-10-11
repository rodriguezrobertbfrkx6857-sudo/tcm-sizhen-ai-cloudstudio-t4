"""Read-only inventory; no optimizer, installation or other-process changes."""
import argparse,json,platform,shutil,subprocess
from pathlib import Path
import torch,torchvision,transformers

def inspect(require_gpu=False):
    z=dict(python=platform.python_version(),platform=platform.system(),torch=torch.__version__,torchvision=torchvision.__version__,transformers=transformers.__version__,CUDA_runtime=torch.version.cuda,CUDA_available=torch.cuda.is_available(),disk_free_GiB=round(shutil.disk_usage('.').free/2**30,2),GPU_checked=False)
    if shutil.which('nvidia-smi'):
        z['nvidia_smi']=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
        z['compute_processes']=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
    if require_gpu:
        if not torch.cuda.is_available():raise RuntimeError('CUDA_UNAVAILABLE')
        z['GPU_checked']=True;z['device']=torch.cuda.get_device_name(0);z['capability']=list(torch.cuda.get_device_capability(0));z['wheel_architectures']=torch.cuda.get_arch_list()
        free,total=torch.cuda.mem_get_info();z['GPU_free_GiB']=free/2**30;z['GPU_total_GiB']=total/2**30
        if z['capability']!=[7,5] or 'T4' not in z['device']:raise RuntimeError('NOT_EXPECTED_T4_VERIFY_NEW_PROFILE')
        if 'sm_75' not in z['wheel_architectures']:raise RuntimeError('WHEEL_MISSING_SM75')
        if free<6*2**30:raise RuntimeError('INSUFFICIENT_FREE_GPU_MEMORY')
        if z.get('compute_processes'):raise RuntimeError('OTHER_COMPUTE_PROCESS_PRESENT_DO_NOT_STOP_IT')
    if z['disk_free_GiB']<20 and require_gpu:raise RuntimeError('REQUIRE_20_GIB_FREE_STORAGE')
    z['status']='GPU_ENVIRONMENT_PASS_NOT_MODEL_PASS' if require_gpu else 'CPU_INVENTORY_ONLY'
    return z

def main():
    p=argparse.ArgumentParser();p.add_argument('--require-gpu',action='store_true');p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('NO_OVERWRITE_REPORT')
    z=inspect(a.require_gpu);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(z,ensure_ascii=False,indent=2),'utf8');print(json.dumps(z,ensure_ascii=False))
if __name__=='__main__':main()
