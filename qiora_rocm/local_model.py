"""Optional future read-only 8B loader. Not called by CPU preparation."""
import json
from pathlib import Path
from scripts.amd_common import require_hip
from .feasibility import parent_manifest

def load_verified_parent(parent_dir, manifest, *, user_approved_load=False,
                         no_other_compute_confirmed=False, min_free_gib=32,
                         attention='eager', precision='fp32'):
    # This function only loads a local parent. No optimizer or training API.
    if not user_approved_load:raise ValueError('NEW_PARENT_LOAD_APPROVAL_REQUIRED')
    if attention not in ('eager','sdpa') or precision not in ('fp32','bf16'):raise ValueError('UNVERIFIED_OPTIMIZATION_FORBIDDEN')
    if min_free_gib<32:raise ValueError('INITIAL_PARENT_MEMORY_RESERVE_32_GIB')
    import torch
    hardware=require_hip(torch,no_other_compute_confirmed,min_free_gib)
    spec,_=parent_manifest(parent_dir,manifest)
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from .vendor import QioraPro,ProConfig
    dtype=torch.float32 if precision=='fp32' else torch.bfloat16
    # No trust_remote_code, network fallback, base download or LoRA merge.
    base=AutoModelForCausalLM.from_pretrained(str(parent_dir),local_files_only=True,trust_remote_code=False,torch_dtype=dtype,attn_implementation=attention).to('cuda')
    pro=QioraPro(base,ProConfig(layers=list(range(18,36)),rank=8,alpha=16,dropout=.05,router_hidden=64,bge_dim=512)).to('cuda')
    for n,p in pro.named_parameters():
        if 'lora_' in n or n.startswith('router.'):p.data=p.data.float()
    pro.set_phase('INFERENCE');pro.eval()
    tok=AutoTokenizer.from_pretrained(str(parent_dir),local_files_only=True,trust_remote_code=False)
    return pro,tok,dict(parent_id_windows=spec['parent_id_windows'],hardware=hardware,
                        model_kind='FROZEN_PARENT_WITH_FRESH_UNTRAINED_EXPERTS',trained_pro=False,
                        Case=False,Semantic=False,precision=precision,attention=attention)
