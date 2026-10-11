"""Capability activation needs hashed real training/data/evaluation evidence."""
import json
from pathlib import Path
from .checkpoint import sha256

def proof_passes(capability,kind):
    try:
        proofs=capability['proofs']
        values={}
        for name in ('data','training','evaluation'):
            item=proofs[name]
            if sha256(item['path'])!=item['sha256']:return False
            values[name]=json.loads(Path(item['path']).read_text(encoding='utf8'))
        if any(x.get('scope')!='REAL_QUALIFIED' or x.get('capability')!=kind for x in values.values()):return False
        data,training,evaluation=(values[k] for k in ('data','training','evaluation'))
        if data.get('eligible_count',0)<=0 or data.get('rights_status')!='CONFIRMED' or data.get('privacy_pass') is not True:return False
        if training.get('status')!='TRAINING_COMPLETE' or training.get('process_exit')!=0:return False
        checkpoint=capability['checkpoint_path'];digest=sha256(checkpoint)
        if digest!=capability['checkpoint_sha256'] or digest!=training.get('checkpoint_sha256') or digest!=evaluation.get('checkpoint_sha256'):return False
        if evaluation.get('status')!='PASS' or evaluation.get('independent_eval_pass') is not True or evaluation.get('reliability_non_degradation') is not True:return False
        return True
    except (KeyError,OSError,ValueError,TypeError):return False
