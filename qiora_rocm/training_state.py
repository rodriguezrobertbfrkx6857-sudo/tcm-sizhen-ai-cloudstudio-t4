"""Atomic immutable adapter checkpoints, strict resume and verified backup copies."""
import json
import os
import shutil
import time
from pathlib import Path
import torch
from .vendor.checkpoint import sha256, save_checkpoint, load_checkpoint
from .training_data import digest, safe_path


def atomic(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.partial')
    with tmp.open('w', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def code_identity():
    root = Path(__file__).parent
    return digest({p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob('*.py'))})


def read_owned(path, expected_sha):
    path = safe_path(path)
    if len(expected_sha) != 64 or sha256(path) != expected_sha:
        raise ValueError('CHECKPOINT_SHA_MISMATCH')
    # Only explicitly SHA-bound, user-owned torch payloads; never arbitrary downloads.
    return torch.load(path, map_location='cpu', weights_only=False)


def verify_state(payload, model, identity, order, accumulation):
    if payload.get('identity') != identity or payload.get('config') != model.pro_config.to_dict():
        # JSON turns tuple into list; checkpoint is native config so exact match is required.
        raise ValueError('RESUME_IDENTITY_CONFIG_MISMATCH')
    if payload.get('phase') not in ('T1', 'T2') or payload.get('data_order') != order:
        raise ValueError('RESUME_PHASE_OR_ORDER_MISMATCH')
    steps, offset = payload.get('steps'), payload.get('offset')
    if type(steps) is not int or type(offset) is not int or offset != min(steps * accumulation,len(order)) or not 0 <= offset <= len(order):
        raise ValueError('RESUME_OFFSET_MISMATCH')
    expected = {n: t for n, t in model.state_dict().items() if 'lora_' in n or n.startswith('router.')}
    actual = payload.get('adapter_router_state', {})
    if set(actual) != set(expected) or any(actual[n].shape != t.shape or actual[n].dtype != t.dtype or not torch.isfinite(actual[n]).all() for n, t in expected.items()):
        raise ValueError('INCOMPLETE_OR_NONFINITE_ADAPTERS')
    if not payload.get('optimizer', {}).get('param_groups') or payload.get('scheduler') is None:
        raise ValueError('RESUME_OPTIMIZER_SCHEDULER_MISSING')
    if steps and not payload['optimizer'].get('state'):
        raise ValueError('RESUME_OPTIMIZER_STATE_EMPTY')
    trainable=[p for p in model.parameters() if p.requires_grad]
    groups=payload['optimizer']['param_groups']
    ids=[i for g in groups for i in g['params']]
    if len(ids)!=len(trainable) or len(set(ids))!=len(ids):raise ValueError('RESUME_OPTIMIZER_PARAMETER_ORDER_INVALID')
    for i,t in zip(ids,trainable):
        s=payload['optimizer']['state'].get(i,{})
        if steps and (not {'step','exp_avg','exp_avg_sq'}<=s.keys() or int(s['step'])!=steps):
            raise ValueError('RESUME_OPTIMIZER_STEP_INVALID')
        for k in ('exp_avg','exp_avg_sq'):
            if k in s and (s[k].shape!=t.shape or not torch.isfinite(s[k]).all()):raise ValueError('RESUME_OPTIMIZER_TENSOR_INVALID')
    if payload['scheduler'].get('last_epoch')!=steps:raise ValueError('RESUME_SCHEDULER_STEP_INVALID')
    if 'python_rng' not in payload or not isinstance(payload.get('torch_rng'), torch.Tensor) or payload['torch_rng'].dtype != torch.uint8:
        raise ValueError('RESUME_RNG_MISSING')
    if torch.cuda.is_available():
        rng = payload.get('cuda_rng')
        if not isinstance(rng, list) or len(rng) != torch.cuda.device_count() or any(x.dtype != torch.uint8 or x.ndim != 1 or x.numel() == 0 for x in rng):
            raise ValueError('RESUME_HIP_RNG_MISSING')
    return payload


def save_boundary(run, backup, model, opt, sched, identity, order, step, offset, tokens, extra):
    dest = Path(run) / f'step_{step:06d}.pt'
    if dest.exists():
        raise ValueError('IMMUTABLE_CHECKPOINT_EXISTS')
    h = save_checkpoint(dest, model, opt, identity=identity, data_order=order, steps=step,
                        offset=offset, tokens=tokens, scheduler=sched)
    if extra.get('parent_manifest_sha'):
        payload=read_owned(dest,h)
        payload['rocm_binding']={k:extra[k] for k in ('parent_manifest_sha','parent_id','code_sha','data_sha')}
        tmp=dest.with_suffix('.binding.partial');torch.save(payload,tmp);os.replace(tmp,dest)
        h=sha256(dest);dest.with_suffix('.sha256').write_text(h,encoding='ascii')
    # Non-model bookkeeping does not alter the vendor payload or its SHA.
    sidecar = dict(extra, checkpoint_sha256=h, steps=step, offset=offset, tokens=tokens, identity=identity)
    atomic(dest.with_suffix('.receipt.json'), sidecar)
    target = Path(backup) / Path(run).name
    target.mkdir(parents=True, exist_ok=True)
    files = [dest, dest.with_suffix('.sha256'), dest.with_suffix('.receipt.json')]
    if (Path(run)/'FROZEN_RUN.json').exists(): files.append(Path(run)/'FROZEN_RUN.json')
    for p in files:
        q = target / p.name
        if q.exists():
            if sha256(q) != sha256(p):
                raise ValueError('BACKUP_COLLISION')
        else:
            temp = q.with_name(q.name + '.partial')
            shutil.copyfile(p, temp)
            with temp.open('r+b') as f: os.fsync(f.fileno())
            os.replace(temp, q)
        if sha256(q) != sha256(p):
            raise ValueError('BACKUP_READBACK_FAILED')
    receipt = dict(sidecar, backup_copy=str(target / dest.name), backup_readback_sha_pass=True,
                   external_location_independently_verified=False,
                   full_external_model_reload='PENDING', timestamp=time.time())
    atomic(Path(run) / 'LATEST.json', receipt)
    return receipt


def restore(path, h, model, opt, sched, identity, order, accumulation):
    p = verify_state(read_owned(path, h), model, identity, order, accumulation)
    loaded = load_checkpoint(path, model, opt, identity=identity, expected_sha256=h, scheduler=sched, restore_rng=True)
    if not torch.equal(torch.get_rng_state(), p['torch_rng']):
        raise ValueError('TORCH_RNG_RESTORE_FAILED')
    if torch.cuda.is_available() and any(not torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), p['cuda_rng'])):
        raise ValueError('HIP_NATIVE_RNG_RESTORE_FAILED')
    return loaded


def require_parent_binding(cp, manifest_sha):
    if cp.get('rocm_binding',{}).get('parent_manifest_sha')!=manifest_sha:
        raise ValueError('EXPERT_NOT_BOUND_TO_THIS_PARENT_MANIFEST')


class RunLock:
    """Single trainer per assets workspace; stale ownership requires explicit inspection."""
    def __init__(self, path): self.path = Path(path)
    def __enter__(self):
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, 'w') as f: json.dump(dict(pid=os.getpid(), started=time.time()), f)
        return self
    def __exit__(self, *args): self.path.unlink()
