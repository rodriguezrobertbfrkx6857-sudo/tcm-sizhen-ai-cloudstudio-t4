"""Real full-parent + three-expert CPU/GPU reload; never a SHA-only restore claim."""
import argparse,os,time
import torch
from qiora_rocm.training_state import read_owned,require_parent_binding,atomic
from qiora_rocm.training_data import safe_path
from qiora_rocm.vendor.checkpoint import sha256
from qiora_rocm.dev_eval import generate
from .train import load,max_code

def install_reader(model,cp,manifest):
    require_parent_binding(cp,sha256(manifest))
    if cp['phase']!='R1' or cp['config']!=model.pro_config.to_dict():raise ValueError('MAX_READER_CONFIG_MISMATCH')
    expected={n:t for n,t in model.state_dict().items() if '.reading.' in n and 'lora_' in n}
    chosen={n:t for n,t in cp['adapter_router_state'].items() if n in expected}
    if set(expected)!=set(chosen) or any(t.shape!=expected[n].shape or t.dtype!=expected[n].dtype or not torch.isfinite(t).all() for n,t in chosen.items()):raise ValueError('INCOMPLETE_READER_STATE')
    model.load_state_dict(chosen,strict=False);model.eval()

def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--parent-manifest',default='configs/qiora_parent_assets_manifest.json')
    for key in ('domain','domain-sha','evidence','evidence-sha','reader','reader-sha','out'):p.add_argument('--'+key,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--confirm-full-reload',action='store_true');a=p.parse_args()
    if not a.confirm_full_reload:raise ValueError('EXPLICIT_FULL_RELOAD_REQUIRED')
    out=safe_path(a.out)
    if out.exists():raise ValueError('IMMUTABLE_PROOF_EXISTS')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';t=time.perf_counter()
    m,tok,_=load(a.parent,a.parent_manifest,a.domain,a.domain_sha,a.evidence,a.evidence_sha,a.device)
    install_reader(m,read_owned(a.reader,a.reader_sha),a.parent_manifest)
    result=generate(m,tok,dict(question='只输出5。'),[.4,.4,.2],a.device,96)
    if not result['raw'].strip() or not result['generated_tokens']:raise ValueError('EMPTY_RESTORE_GENERATION')
    atomic(out,dict(status='ACTUAL_FULL_PARENT_THREE_EXPERT_RELOAD_GENERATION_PASS',device=a.device,code_sha=max_code(),parent_sha=sha256(a.parent_manifest),domain_sha=a.domain_sha,evidence_sha=a.evidence_sha,reader_sha=a.reader_sha,result=result,seconds=time.perf_counter()-t,Max_accepted=False,optimizer_resume_tested=False,HIP_RNG_resume_tested=False))

if __name__=='__main__':main()
