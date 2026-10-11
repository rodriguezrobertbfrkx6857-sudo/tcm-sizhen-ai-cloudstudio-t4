import hashlib
import json
import os
import random
from pathlib import Path
import torch

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def save_checkpoint(path, model, optimizer, *, identity, offset, steps, data_order, scheduler=None, tokens=0):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    # Frozen 8B parent is never copied into adapter checkpoints.
    state = {n: t.detach().cpu() for n, t in model.state_dict().items() if 'lora_' in n or n.startswith('router.')}
    payload = {'schema': 2, 'identity': identity, 'config': model.pro_config.to_dict(), 'phase': model.phase,
               'adapter_router_state': state, 'optimizer': optimizer.state_dict(), 'offset': offset,
               'steps': steps, 'data_order': data_order, 'tokens': tokens,
               'scheduler': scheduler.state_dict() if scheduler is not None else None, 'python_rng': random.getstate(),
               'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    tmp = path.with_suffix(path.suffix + '.partial')
    torch.save(payload, tmp); os.replace(tmp, path)
    digest = sha256(path)
    path.with_suffix('.sha256').write_text(digest, encoding='ascii')
    return digest

def load_checkpoint(path, model, optimizer=None, *, identity, expected_sha256, restore_rng=True, scheduler=None):
    if sha256(path) != expected_sha256:
        raise ValueError('Checkpoint SHA mismatch')
    p = torch.load(path, map_location='cpu', weights_only=False)
    if p['identity'] != identity or fingerprint(p['config']) != fingerprint(model.pro_config.to_dict()):
        raise ValueError('Checkpoint lineage/config identity mismatch')
    required = {n for n in model.state_dict() if 'lora_' in n or n.startswith('router.')}
    if set(p['adapter_router_state']) != required:
        raise ValueError('Incomplete/incompatible expert checkpoint')
    result = model.load_state_dict(p['adapter_router_state'], strict=False)
    if result.unexpected_keys:
        raise ValueError('Unexpected state keys')
    model.set_phase(p['phase'], case_data_pass=p['phase'] == 'T5')
    model.loaded_checkpoint_sha256 = expected_sha256
    if optimizer is not None:
        optimizer.load_state_dict(p['optimizer'])
    if scheduler is not None:
        if p.get('scheduler') is None: raise ValueError('Resume lacks frozen scheduler state')
        scheduler.load_state_dict(p['scheduler'])
    if restore_rng:
        random.setstate(p['python_rng']); torch.set_rng_state(p['torch_rng'])
        if p['cuda_rng'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(p['cuda_rng'])
    return p
