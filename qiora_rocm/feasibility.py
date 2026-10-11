"""Native Qiora structural probes. Zero optimizer updates, no remote models."""
import argparse,copy,importlib.metadata,json,os,random,tempfile,time
from pathlib import Path
from scripts.amd_common import finish_report,require_hip,sha256
from scripts.asset_verify import verify

def tiny(device='cpu',confirmed=False,attention='eager',precision='fp32'):
    import torch
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from .vendor import QioraPro,ProConfig
    from .vendor.checkpoint import save_checkpoint,load_checkpoint
    if device=='cuda':hardware=require_hip(torch,confirmed)
    else:
        hardware=dict(device='LOCAL_CPU_NO_ROCM_VERIFICATION')
        if precision!='fp32':raise ValueError('CPU_PROOF_ONLY_FP32')
    if precision not in ('fp32','bf16'):raise ValueError('UNSUPPORTED_PRECISION')
    dtype=torch.float32 if precision=='fp32' else torch.bfloat16
    torch.set_num_threads(2);random.seed(20261011);torch.manual_seed(20261011)
    if device=='cuda':torch.cuda.manual_seed_all(20261011);torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    c=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=64,bos_token_id=1,eos_token_id=2,pad_token_id=0,tie_word_embeddings=False)
    c._attn_implementation=attention
    model=QioraPro(Qwen3ForCausalLM(c).to(dtype=dtype),ProConfig(layers=[1],rank=2,alpha=4,dropout=0.,router_hidden=8,bge_dim=4)).to(device)
    # Match existing production policy: adapter and router tensors stay FP32.
    for n,p in model.named_parameters():
        if 'lora_' in n or n.startswith('router.'):p.data=p.data.float()
    model.backbone.config.use_cache=False
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
    ids=torch.tensor([[1,4,5,6,7,8,9,10]],device=device);labels=ids.clone();labels[:,:3]=-100
    mask=torch.ones_like(ids);phases=[]
    for phase in ['T1','T2']:
        model.set_phase(phase);model.eval();model.zero_grad(set_to_none=True)
        output,w=model(ids,attention_mask=mask,labels=labels);loss=output.loss;loss.backward()
        grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        norm=float(sum(g.float().square().sum() for g in grads).sqrt())
        frozen_ok=all(p.grad is None for p in model.parameters() if not p.requires_grad)
        if not torch.isfinite(loss) or not frozen_ok or not norm>0 or not all(torch.isfinite(g).all() for g in grads):raise ValueError('EXPERT_GRADIENT_OR_FREEZE_FAILURE')
        phases.append(dict(phase=phase,loss=float(loss.detach()),gradient_norm=norm,frozen_parameter_gradients_absent=frozen_ok,active_mask=list(model.active_mask),trainable_parameters=model.parameter_audit()['trainable_parameters']))
    model.set_phase('T3');model.eval();weights=model.coefficients(ids,mask,[dict(intent='fact',source_type='none',mode='normal')]);(-weights[:,0].clamp_min(1e-8).log().mean()).backward()
    if weights[0,2].item()!=0 or not any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if n.startswith('router.')):raise ValueError('ROUTER_GRADIENT_OR_CASE_MASK_FAILURE')
    if any(p.grad is not None for n,p in model.named_parameters() if not n.startswith('router.')):raise ValueError('ROUTER_FREEZE_FAILURE')
    model.set_phase('INFERENCE');model.eval()
    with torch.no_grad():before=model(ids,attention_mask=mask)[0].logits.detach().cpu()
    rng_before=torch.get_rng_state().clone()
    # Constructors only; no optimizer.step(), scheduler.step(), or parameter update.
    opt=torch.optim.AdamW(model.router.parameters(),lr=1e-4);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
    with tempfile.TemporaryDirectory(prefix='qiora_random_fixture_') as td:
        dest=Path(td)/'tiny_owned_state.pt';digest=save_checkpoint(dest,model,opt,identity='RANDOM_TINY_FIXTURE_NOT_PRO',offset=0,steps=0,data_order=[],scheduler=sched)
        with torch.no_grad():next(model.router.parameters()).add_(1) # disposable restore challenge only
        state=load_checkpoint(dest,model,opt,identity='RANDOM_TINY_FIXTURE_NOT_PRO',expected_sha256=digest,scheduler=sched,restore_rng=True)
        with torch.no_grad():after=model(ids,attention_mask=mask)[0].logits.detach().cpu()
        if not torch.equal(before,after) or not torch.equal(rng_before,torch.get_rng_state()) or state['steps']!=0:raise ValueError('RESTORE_LOGITS_OR_RNG_FAILURE')
    if not all(torch.equal(initial[n],p.detach().cpu()) for n,p in model.named_parameters()):raise ValueError('PARAMETER_MUTATION_WITHOUT_RESTORE')
    if device=='cuda':torch.cuda.synchronize()
    return dict(status='PASS',measurement='TINY_RANDOM_QWEN3_NATIVE_QIORA_STRUCTURE_NOT_8B',hardware=hardware,
                attention=attention,precision=precision,torch=torch.__version__,transformers=importlib.metadata.version('transformers'),peft=importlib.metadata.version('peft'),phases=phases,
                router_weights=weights.detach().cpu().tolist(),router_gradient_pass=True,case_active=False,semantic_enabled=False,
                parent_parameters_unchanged=True,restore_logits_exact=True,python_torch_rng_state_present=True,CPU_RNG_restored=True,
                gpu_rng_restore_tested=device=='cuda',optimizer_state_empty_no_updates=True,synthetic_checkpoint_retained=False,
                seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None)

def parent_manifest(root,manifest):
    spec=json.loads(Path(manifest).read_text('utf8'))
    if spec.get('format')!='HF_BF16_MERGED_QIORA' or len(spec.get('assets',[]))!=14:raise ValueError('NOT_FROZEN_QIORA_PARENT_MANIFEST')
    r=verify(root,manifest)
    if r['status']!='PASS':raise ValueError('PARENT_FILES_FAILED_SHA')
    c=json.loads((Path(root)/'config.json').read_text('utf8'))
    if c.get('model_type')!='qwen3' or c.get('num_hidden_layers')!=36 or c.get('hidden_size')!=4096:raise ValueError('NOT_EXPECTED_QIORA_ARCHITECTURE')
    return spec,r

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--confirm-no-other-compute',action='store_true');p.add_argument('--attention',choices=['eager','sdpa'],default='eager');p.add_argument('--precision',choices=['fp32','bf16'],default='fp32');p.add_argument('--verify-parent-only');p.add_argument('--parent-manifest',default='configs/qiora_parent_assets_manifest.json');p.add_argument('--out');a=p.parse_args()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    try:
        if a.verify_parent_only:
            _,r=parent_manifest(a.verify_parent_only,a.parent_manifest);r=dict(r,measurement='LOCAL_PARENT_SHA_ONLY_NOT_FULL_MODEL_RELOAD')
        else:r=tiny(a.device,a.confirm_no_other_compute,a.attention,a.precision)
    except Exception as exc:r=dict(status='FAIL',error_type=type(exc).__name__,measurement='NO_SUCCESS_CLAIM')
    finish_report(r,a.out);return 0 if r['status']=='PASS' else 2
if __name__=='__main__':raise SystemExit(main())
