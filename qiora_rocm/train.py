"""ROCm Pro expert trainer. Invoke with python -m qiora_rocm.train."""
import argparse
import json
import math
import os
import random
import signal
import socket
import time
import uuid
from pathlib import Path
import torch
from .training_data import rows, isolate, encode, digest, safe_path
from .training_state import atomic, code_identity, RunLock, restore, save_boundary, read_owned
from .vendor import QioraPro, ProConfig
from .vendor.checkpoint import sha256


def load_model(parent, manifest, device='cuda'):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .feasibility import parent_manifest
    spec, _ = parent_manifest(parent, manifest)
    if device == 'cuda' and (not torch.version.hip or not torch.cuda.is_available()):
        raise ValueError('ROCM_GPU_REQUIRED')
    if device == 'cuda' and torch.cuda.mem_get_info()[0] < 32 * 1024**3:
        raise ValueError('TRAIN_LOAD_NEEDS_32_GIB_REAL_FREE_VRAM')
    base = AutoModelForCausalLM.from_pretrained(str(parent), local_files_only=True, trust_remote_code=False,
                                               torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
                                               attn_implementation='sdpa').to(device)
    pro = QioraPro(base, ProConfig(layers=list(range(18, 36)), rank=8, alpha=16, dropout=.05)).to(device)
    for n, p in pro.named_parameters():
        if 'lora_' in n or n.startswith('router.'):
            p.data = p.data.float()
    pro.backbone.config.use_cache = False
    tok = AutoTokenizer.from_pretrained(str(parent), local_files_only=True, trust_remote_code=False)
    return pro, tok, spec


def update(model, encoded, weights, optimizer, scheduler, device, clip=1.):
    """One optimizer boundary; row means weighted then averaged across accumulation."""
    model.train(); optimizer.zero_grad(set_to_none=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    frozen = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
    losses = []
    for x, weight in zip(encoded, weights):
        ids = torch.tensor([x['input_ids']], device=device)
        labels = torch.tensor([x['labels']], device=device)
        loss = model(ids, attention_mask=torch.ones_like(ids), labels=labels)[0].loss
        if not torch.isfinite(loss): raise ValueError('NONFINITE_LOSS')
        (loss * weight / len(encoded)).backward()
        losses.append(float(loss.detach()))
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise ValueError('FROZEN_PARAMETER_GRADIENT')
    norm = float(torch.nn.utils.clip_grad_norm_(trainable, clip, error_if_nonfinite=True))
    if not math.isfinite(norm) or norm <= 0: raise ValueError('ZERO_OR_NONFINITE_GRADIENT')
    optimizer.step(); scheduler.step()
    if any(p._version != frozen[n] for n, p in model.named_parameters() if n in frozen):
        raise ValueError('FROZEN_PARAMETER_MUTATED')
    return dict(losses=losses, weighted_loss=sum(x*w for x,w in zip(losses, weights))/len(weights),
                gradient_norm=norm, frozen_parameter_versions_unchanged=True)


def train(a):
    request_started = time.time()
    if not a.confirm_training: raise ValueError('EXPLICIT_TRAINING_FLAG_REQUIRED')
    if not torch.version.hip or not torch.cuda.is_available(): raise ValueError('ROCM_REQUIRED_NO_CPU_FORMAL_TRAINING')
    if not (0 < a.max_steps <= 64 and 0 < a.stop_after <= a.max_steps and 0 < a.max_seconds <= 28800):
        raise ValueError('INITIAL_BOUNDED_64_STEPS_8H_MAX')
    if a.accumulation != 4 or a.seq_len != 2048 or a.save_every not in (1, 2, 4, 8):
        raise ValueError('INITIAL_FROZEN_SCHEDULE_4_MICROBATCH_2048_REQUIRED')
    manifest = json.loads(safe_path(a.data_manifest).read_text('utf8'))
    root = safe_path(a.data_dir); phase = 'T1' if a.expert == 'domain' else 'T2'
    key = 'D_REPAIR' if a.expert == 'domain' else 'E_COMPACT'
    spec = manifest['datasets'][key]
    rr = rows(root / spec['file'], spec['sha256'], 'TRAIN', spec.get('row_selection'))
    dev = []
    for k in manifest['dev_sets']:
        s = manifest['datasets'][k]; dev += rows(root / s['file'], s['sha256'], 'DEV')
    isolation = isolate(rr, dev)
    if len(rr) != spec['selected_rows'] or a.max_steps > math.ceil(len(rr)/a.accumulation): raise ValueError('STEP_DATA_EXHAUSTION')
    run = safe_path(a.run_dir); backup = safe_path(a.backup_dir)
    if run == backup or backup.is_relative_to(run) or run.is_relative_to(backup): raise ValueError('SEPARATE_BACKUP_PATH_REQUIRED')
    if a.resume and not a.resume_sha: raise ValueError('EXPLICIT_RESUME_SHA_REQUIRED')
    if run.exists() and not a.resume: raise ValueError('EXISTING_RUN_NEEDS_EXPLICIT_RESUME')
    if not run.exists() and a.resume: raise ValueError('MISSING_RUN_BINDING')
    backup.mkdir(parents=True, exist_ok=True)
    run.mkdir(parents=True, exist_ok=True)
    import shutil
    if min(shutil.disk_usage(run).free,shutil.disk_usage(backup).free)<2*1024**3:
        raise ValueError('CHECKPOINT_BACKUP_DISK_RESERVE_2_GIB_REQUIRED')
    stop = {'requested': False}
    def request_stop(signum, frame): stop['requested'] = True
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, request_stop)
    attempt = uuid.uuid4().hex
    with RunLock(safe_path(a.parent).parent / 'QIORA_TRAINER.lock'):
        random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
        torch.cuda.reset_peak_memory_stats()
        model, tok, parent = load_model(safe_path(a.parent), a.parent_manifest)
        enc = [encode(tok, r, a.seq_len) for r in rr]
        if any(x['input_tokens'] != r['input_tokens'] or x['supervised_tokens'] != r['supervised_tokens'] for x,r in zip(enc,rr)):
            raise ValueError('FROZEN_TOKENIZER_FINGERPRINT_MISMATCH')
        order = [r['sample_id'] for r in rr]
        plan = dict(expert=a.expert, phase=phase, max_steps=a.max_steps, accumulation=a.accumulation,
                    seq_len=a.seq_len, learning_rate=a.lr or spec['learning_rate'], save_every=a.save_every,
                    seed=a.seed, parent_manifest_sha=sha256(a.parent_manifest), parent_id=parent['parent_id_windows'],
                    data_sha=spec['sha256'], data_manifest_sha=sha256(a.data_manifest), order_sha=digest(order),
                    tokenization_sha=digest(enc), code_sha=code_identity(), max_seconds=a.max_seconds,
                    torch_version=torch.__version__, hip_version=torch.version.hip,
                    package_versions={k: __import__('importlib.metadata', fromlist=['version']).version(k) for k in ('transformers','peft')},
                    architecture=model.pro_config.to_dict(), Case=False, Semantic=False, Router_updates=0)
        # Normalize tuple fields identically on disk and at restart.
        plan = json.loads(json.dumps(plan))
        if a.resume:
            frozen = json.loads((run/'FROZEN_RUN.json').read_text('utf8'))
            if frozen['plan'] != plan: raise ValueError('FROZEN_RUN_CHANGED')
        else:
            frozen = dict(plan=plan, nonce=uuid.uuid4().hex, run_id=run.name,
                          start_unix=request_started, hostname=socket.gethostname(), isolation=isolation,
                          external_backup_verified=False, independent_acceptance=False)
            atomic(run/'FROZEN_RUN.json', frozen)
        identity = digest(frozen)
        deadline = frozen['start_unix'] + a.max_seconds
        if time.time() >= deadline - 120: raise ValueError('CUMULATIVE_WINDOW_EXHAUSTED')
        model.set_phase(phase)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=plan['learning_rate'], weight_decay=0.)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda n: 1.)
        steps = offset = tokens = 0
        if a.resume:
            p = restore(a.resume, a.resume_sha, model, opt, sched, identity, order, a.accumulation)
            steps, offset, tokens = p['steps'], p['offset'], p['tokens']
            if p['phase'] != phase or tokens != sum(x['input_tokens'] for x in enc[:offset]): raise ValueError('PHASE_OR_TOKEN_OFFSET_MISMATCH')
            # Preserve old attempt logs, including updates beyond last durable checkpoint.
            atomic(run/f'RESUME_{attempt}.json', dict(from_path=a.resume, checkpoint_sha=a.resume_sha,
                   durable_step=steps, old_attempts_preserved=True, old_uncommitted_updates_not_relabelled=True))
        audit = model.parameter_audit()
        if not audit['whitelist'] or any(f'.{a.expert}.' not in n for n in audit['whitelist']): raise ValueError('TRAINABLE_WHITELIST_FAILURE')
        atomic(run/f'PARAMETER_AUDIT_{attempt}.json', audit)
        def save():
            return save_boundary(run, backup, model, opt, sched, identity, order, steps, offset, tokens,
                                 dict(attempt=attempt, supervised_tokens=sum(x['supervised_tokens'] for x in enc[:offset]),
                                      started_unix=frozen['start_unix'], deadline_unix=deadline, code_sha=plan['code_sha'],
                                      parent_manifest_sha=plan['parent_manifest_sha'], parent_id=plan['parent_id'], data_sha=plan['data_sha']))
        if not a.resume: save()
        status = 'STOPPED_AT_REQUESTED_BOUNDARY'
        try:
            while steps < a.stop_after:
                if stop['requested'] or time.time() > deadline - 120: break
                t = time.perf_counter(); batch = enc[offset:offset+a.accumulation]
                result = update(model, batch, [r.get('loss_weight',1.) for r in rr[offset:offset+a.accumulation]], opt, sched, 'cuda')
                torch.cuda.synchronize(); steps += 1; offset += len(batch)
                tokens = sum(x['input_tokens'] for x in enc[:offset])
                free, total = torch.cuda.mem_get_info()
                result.update(step=steps, offset=offset, input_tokens=tokens, attempt=attempt, nonce=frozen['nonce'],
                              seconds=time.perf_counter()-t, tokens_per_second=sum(x['input_tokens'] for x in batch)/(time.perf_counter()-t),
                              peak_allocated_bytes=torch.cuda.max_memory_allocated(), free_vram_bytes=free,
                              total_vram_bytes=total, timestamp=time.time())
                with (run/f'UPDATES_{attempt}.jsonl').open('a',encoding='utf8') as f:
                    f.write(json.dumps(result,allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
                print(json.dumps(result), flush=True)
                if steps % a.save_every == 0 or steps == a.stop_after: save()
            if steps == a.max_steps: status = 'TRAINED_PENDING_DEV_REVIEW'
            if not (run/f'step_{steps:06d}.pt').exists(): save()
            # Real native restore at last optimizer boundary, same saved RNG.
            latest = json.loads((run/'LATEST.json').read_text('utf8'))
            model.eval()
            challenge = torch.tensor([enc[0]['input_ids']], device='cuda')
            with torch.no_grad():before = model(challenge)[0].logits.detach().cpu()
            restore(run/f'step_{steps:06d}.pt', latest['checkpoint_sha256'], model, opt, sched, identity, order, a.accumulation)
            with torch.no_grad():after = model(challenge)[0].logits.detach().cpu()
            if not torch.equal(before, after):raise ValueError('RESTORE_LOGITS_MISMATCH')
            atomic(run/f'RESTORE_{attempt}.json', dict(status='OPTIMIZER_SCHEDULER_PYTHON_TORCH_HIP_RNG_RELOADED',
                   steps=steps, checkpoint_sha=latest['checkpoint_sha256'], logits_equal=True, full_external_reload=False))
        except BaseException as exc:
            # Never checkpoint ambiguous partial optimizer state; the latest committed boundary survives.
            atomic(run/f'FAILURE_{attempt}.json', dict(error_type=type(exc).__name__, message=str(exc),
                   last_durable=json.loads((run/'LATEST.json').read_text('utf8')), explicit_resume_required=True))
            raise
        atomic(run/f'STATUS_{attempt}.json',dict(status=status, optimizer_updates=steps, expert=a.expert,
               external_full_restore='PENDING', pro_accepted=False, TEST2='SEALED'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', required=True); p.add_argument('--parent-manifest', default='configs/qiora_parent_assets_manifest.json')
    p.add_argument('--data-dir', required=True); p.add_argument('--data-manifest', default='configs/pro_revision03_data_manifest.json')
    p.add_argument('--run-dir', required=True); p.add_argument('--backup-dir', required=True)
    p.add_argument('--expert', choices=['domain','evidence'], required=True)
    p.add_argument('--max-steps',type=int,default=64); p.add_argument('--stop-after',type=int,default=64)
    p.add_argument('--accumulation',type=int,default=4); p.add_argument('--seq-len',type=int,default=2048)
    p.add_argument('--lr',type=float); p.add_argument('--save-every',type=int,default=8)
    p.add_argument('--seed',type=int,default=20261013); p.add_argument('--max-seconds',type=int,default=7200)
    p.add_argument('--resume'); p.add_argument('--resume-sha'); p.add_argument('--confirm-training',action='store_true')
    a=p.parse_args()
    if a.lr is not None and not 0 < a.lr <= 5e-5: p.error('initial learning rate must be <=5e-5')
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    train(a)


if __name__ == '__main__': main()
