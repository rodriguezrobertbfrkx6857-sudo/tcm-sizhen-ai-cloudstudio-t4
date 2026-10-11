"""Bounded reader-expert training after actual Pro checkpoint availability."""
import argparse
import json
import math
import os
import random
import signal
import time
import uuid
from pathlib import Path
import torch
from peft.tuners.lora.layer import Linear
from qiora_rocm.vendor import ProConfig
from qiora_rocm.vendor.checkpoint import sha256,load_checkpoint
from qiora_rocm.feasibility import parent_manifest
from qiora_rocm.training_data import rows,encode,isolate,digest,safe_path
from qiora_rocm.training_state import atomic,code_identity,RunLock,read_owned,verify_state,save_boundary,require_parent_binding
from qiora_rocm.train import update
from qiora_rocm.dev_eval import fuse
from .model import QioraMax


def max_code():return digest(dict(pro=code_identity(),max={p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}))


def load(parent,manifest,domain,dsha,evidence,esha,device='cuda'):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    spec,_=parent_manifest(parent,manifest)
    if device=='cuda':
        if not torch.version.hip or not torch.cuda.is_available():raise ValueError('ROCM_REQUIRED')
        if torch.cuda.mem_get_info()[0]<32*1024**3:raise ValueError('FREE_VRAM_32_GIB_REQUIRED')
    base=AutoModelForCausalLM.from_pretrained(parent,local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16 if device=='cuda' else torch.float32,attn_implementation='sdpa').to(device)
    model=QioraMax(base,ProConfig(layers=list(range(18,36)))).to(device)
    for n,p in model.named_parameters():
        if 'lora_' in n or n.startswith('router.'):p.data=p.data.float()
    dc=read_owned(domain,dsha);ec=read_owned(evidence,esha)
    for cp in (dc,ec):require_parent_binding(cp,sha256(manifest))
    fuse(model,dc,ec);model.set_phase('R1');model.backbone.config.use_cache=False
    tok=AutoTokenizer.from_pretrained(parent,local_files_only=True,trust_remote_code=False)
    return model,tok,spec


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',required=True);p.add_argument('--parent-manifest',default='configs/qiora_parent_assets_manifest.json')
    p.add_argument('--domain',required=True);p.add_argument('--domain-sha',required=True);p.add_argument('--evidence',required=True);p.add_argument('--evidence-sha',required=True)
    p.add_argument('--data-dir',required=True);p.add_argument('--run-dir',required=True);p.add_argument('--backup-dir',required=True)
    p.add_argument('--max-steps',type=int,default=32);p.add_argument('--stop-after',type=int,default=2);p.add_argument('--max-seconds',type=int,default=3600)
    p.add_argument('--resume');p.add_argument('--resume-sha');p.add_argument('--confirm-research-training',action='store_true');a=p.parse_args()
    if not a.confirm_research_training or not 0<a.stop_after<=a.max_steps<=64 or not 0<a.max_seconds<=7200:raise ValueError('BOUNDED_RESEARCH_CONFIRMATION_REQUIRED')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    started=time.time();data=safe_path(a.data_dir);m=json.loads((data/'MANIFEST.json').read_text('utf8'))
    rr=rows(data/m['datasets']['R_TRAIN']['file'],m['datasets']['R_TRAIN']['sha256'],'TRAIN');dev=rows(data/m['datasets']['R_DEV']['file'],m['datasets']['R_DEV']['sha256'],'DEV');isolate(rr,dev)
    if a.max_steps>math.ceil(len(rr)/4):raise ValueError('READER_DATA_EXHAUSTION')
    run=safe_path(a.run_dir);backup=safe_path(a.backup_dir)
    if run==backup or run.is_relative_to(backup) or backup.is_relative_to(run):raise ValueError('SEPARATE_BACKUP_REQUIRED')
    if run.exists() and not a.resume:raise ValueError('EXPLICIT_RESUME_REQUIRED')
    if a.resume and (not run.exists() or not a.resume_sha):raise ValueError('MISSING_RESUME_BINDING')
    run.mkdir(parents=True,exist_ok=True);backup.mkdir(parents=True,exist_ok=True);stop=[False]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    with RunLock(safe_path(a.parent).parent/'QIORA_TRAINER.lock'):
        random.seed(20261013);torch.manual_seed(20261013);torch.cuda.manual_seed_all(20261013)
        model,tok,parent=load(a.parent,a.parent_manifest,a.domain,a.domain_sha,a.evidence,a.evidence_sha)
        encoded=[encode(tok,r,2048) for r in rr];order=[r['sample_id'] for r in rr]
        plan=dict(parent_manifest_sha=sha256(a.parent_manifest),domain_sha=a.domain_sha,evidence_sha=a.evidence_sha,
                  max_code_sha=max_code(),data_sha=sha256(data/'MANIFEST.json'),tokenization_sha=digest(encoded),max_steps=a.max_steps,
                  max_seconds=a.max_seconds,lr=2e-5,accumulation=4,torch=torch.__version__,hip=torch.version.hip)
        if a.resume:
            frozen=json.loads((run/'FROZEN_RUN.json').read_text('utf8'))
            if frozen['plan']!=plan:raise ValueError('MAX_RUN_IDENTITY_CHANGED')
        else:
            frozen=dict(plan=plan,nonce=uuid.uuid4().hex,start_unix=started,architecture='DOMAIN_EVIDENCE_READING_PROMPT_LAYER_ROUTE_RESEARCH',Router_updates=0)
            atomic(run/'FROZEN_RUN.json',frozen)
        identity=digest(frozen);deadline=frozen['start_unix']+a.max_seconds
        opt=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=2e-5,weight_decay=0.);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
        step=offset=tokens=0;attempt=uuid.uuid4().hex
        if a.resume:
            cp=read_owned(a.resume,a.resume_sha)
            if cp['phase']!='R1':raise ValueError('WRONG_MAX_PHASE')
            checked=dict(cp,phase='T2');verify_state(checked,model,identity,order,4)
            cp=load_checkpoint(a.resume,model,opt,identity=identity,expected_sha256=a.resume_sha,scheduler=sched,restore_rng=True)
            if any(not torch.equal(x,y) for x,y in zip(torch.cuda.get_rng_state_all(),cp['cuda_rng'])):raise ValueError('HIP_RNG_RESTORE_FAILED')
            step,offset,tokens=cp['steps'],cp['offset'],cp['tokens']
            if tokens!=sum(x['input_tokens'] for x in encoded[:offset]) or step>a.max_steps:raise ValueError('RESUME_TOKEN_BUDGET_MISMATCH')
        def save():return save_boundary(run,backup,model,opt,sched,identity,order,step,offset,tokens,
            dict(parent_manifest_sha=plan['parent_manifest_sha'],parent_id=parent['parent_id_windows'],code_sha=plan['max_code_sha'],data_sha=plan['data_sha'],Max_accepted=False))
        if not a.resume:save()
        try:
            while step<a.stop_after and not stop[0] and time.time()<deadline-120:
                batch=encoded[offset:offset+4];t=time.perf_counter();z=update(model,batch,[1.]*len(batch),opt,sched,'cuda');torch.cuda.synchronize()
                step+=1;offset+=len(batch);tokens=sum(x['input_tokens'] for x in encoded[:offset])
                z.update(step=step,offset=offset,seconds=time.perf_counter()-t,peak_allocated_bytes=torch.cuda.max_memory_allocated(),nonce=frozen['nonce'])
                with (run/f'UPDATES_{attempt}.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(z)+'\n');f.flush();os.fsync(f.fileno())
                if step%8==0 or step==a.stop_after:save()
            if not (run/f'step_{step:06d}.pt').exists():save()
            atomic(run/f'STATUS_{attempt}.json',dict(status='MAX_READER_RESEARCH_PENDING_PAIRED_EVALUATION',steps=step,Router_updates=0,external_full_restore='PENDING',accepted=False))
        except BaseException as exc:
            atomic(run/f'FAILURE_{attempt}.json',dict(error_type=type(exc).__name__,last_durable=json.loads((run/'LATEST.json').read_text('utf8')),explicit_resume_required=True));raise


if __name__=='__main__':main()
