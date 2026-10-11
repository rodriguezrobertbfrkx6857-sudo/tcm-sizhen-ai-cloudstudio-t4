"""Single CPU validation entry. No hardware acceleration or remote calls."""
import argparse,os,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')
    commands=[['-m','unittest','discover','-s','tests','-v'],
              ['scripts/amd_rocm_probe.py','--workspace','.','--out',str(out/'inventory.json')],
              ['scripts/amd_visual_smoke.py','--device','cpu','--out',str(out/'visual.json')],
              ['-m','qiora_rocm.feasibility','--device','cpu','--attention','eager','--out',str(out/'qiora_eager.json')],
              ['-m','qiora_rocm.feasibility','--device','cpu','--attention','sdpa','--out',str(out/'qiora_sdpa.json')]]
    for i,args in enumerate(commands):
        r=subprocess.run([sys.executable,*args],cwd=Path(__file__).resolve().parents[1],env=env,capture_output=True,text=True,timeout=300)
        (out/f'command_{i:02d}.log').write_text(r.stdout+r.stderr,encoding='utf8')
        if r.returncode:print('LOCAL_VALIDATION_FAILED_STEP',i);return r.returncode
    print('LOCAL_CPU_VALIDATION_PASS_NOT_GPU_PROOF');return 0
if __name__=='__main__':raise SystemExit(main())
