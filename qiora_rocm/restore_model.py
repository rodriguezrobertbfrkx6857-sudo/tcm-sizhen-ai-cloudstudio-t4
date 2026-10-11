"""Actual full-parent + downloaded adapter reload and generation, separate from SHA-only backup."""
import argparse
import json
import os
import time
from .training_data import safe_path
from .training_state import read_owned,atomic,code_identity,require_parent_binding
from .train import load_model
from .dev_eval import fuse,generate
from .vendor.checkpoint import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',required=True);p.add_argument('--parent-manifest',default='configs/qiora_parent_assets_manifest.json')
    p.add_argument('--domain');p.add_argument('--domain-sha');p.add_argument('--evidence');p.add_argument('--evidence-sha')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--out',required=True)
    p.add_argument('--confirm-full-reload',action='store_true');a=p.parse_args()
    if not a.confirm_full_reload:raise ValueError('FULL_MODEL_RELOAD_CONFIRMATION_REQUIRED')
    if not (a.domain or a.evidence):raise ValueError('NEW_EXPERT_REQUIRED')
    out=safe_path(a.out)
    if out.exists():raise ValueError('IMMUTABLE_PROOF_EXISTS')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    dc=read_owned(a.domain,a.domain_sha) if a.domain else None;ec=read_owned(a.evidence,a.evidence_sha) if a.evidence else None
    for cp in (dc,ec):
        if cp is not None:require_parent_binding(cp,sha256(a.parent_manifest))
    t=time.perf_counter();model,tok,parent=load_model(safe_path(a.parent),a.parent_manifest,a.device)
    fuse(model,dc,ec);w=[.5,.5,0.] if dc and ec else ([1.,0.,0.] if dc else [0.,1.,0.])
    samples=[]
    for question in ['只输出5。','请用一句话说明中医阴阳概念属于什么知识范围。']:
        samples.append(dict(question=question,**generate(model,tok,dict(question=question),w,a.device,96)))
    if not all(x['generated_tokens']>0 and x['raw'].strip() for x in samples):raise ValueError('EMPTY_RESTORE_GENERATION')
    atomic(out,dict(status='ACTUAL_FULL_PARENT_AND_EXPERT_RELOAD_GENERATION_PASS',device=a.device,
        parent_manifest_sha=sha256(a.parent_manifest),parent_id=parent['parent_id_windows'],code_sha=code_identity(),
        expert_SHAs=dict(domain=a.domain_sha,evidence=a.evidence_sha),samples=samples,seconds=time.perf_counter()-t,
        optimizer_resume_tested=False,HIP_RNG_resume_tested=False,clinical_validation=False,pro_accepted=False))


if __name__=='__main__':main()
