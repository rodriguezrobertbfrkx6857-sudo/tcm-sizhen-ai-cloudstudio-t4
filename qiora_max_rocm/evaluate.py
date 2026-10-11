"""Compare Base/Pro/Max on revealed exact-source-reading DEV, not clinical claims."""
import argparse
import collections
import json
import os
import torch
from qiora_rocm.training_data import rows,safe_path,prompt_ids,digest
from qiora_rocm.training_state import read_owned,require_parent_binding,atomic,RunLock
from qiora_rocm.vendor.checkpoint import sha256
from qiora_rocm.dev_eval import generate
from .train import load,max_code


def score(row,raw):
    try:
        value=json.loads(raw);expected=json.loads(row['answer'])
    except (ValueError,TypeError):return 'INVALID_FORMAT'
    return 'CORRECT' if value==expected else 'INCORRECT'


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--parent-manifest',default='configs/qiora_parent_assets_manifest.json')
    for name in ('domain','domain-sha','evidence','evidence-sha','reader','reader-sha','data-dir','out'):p.add_argument('--'+name,required=True)
    a=p.parse_args();os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    data=safe_path(a.data_dir);spec=json.loads((data/'MANIFEST.json').read_text('utf8'))['datasets']['R_DEV'];rr=rows(data/spec['file'],spec['sha256'],'DEV')
    out=safe_path(a.out);out.mkdir(parents=True,exist_ok=False)
    cp=read_owned(a.reader,a.reader_sha);require_parent_binding(cp,sha256(a.parent_manifest))
    if cp['phase']!='R1':raise ValueError('WRONG_READER_CHECKPOINT')
    with RunLock(safe_path(a.parent).parent/'QIORA_TRAINER.lock'):
        model,tok,_=load(a.parent,a.parent_manifest,a.domain,a.domain_sha,a.evidence,a.evidence_sha)
        if cp['config']!=model.pro_config.to_dict():raise ValueError('MAX_CONFIG_MISMATCH')
        expected={n for n in model.state_dict() if 'lora_A.reading.' in n or 'lora_B.reading.' in n}
        state={n:t for n,t in cp['adapter_router_state'].items() if n in expected}
        if set(state)!=expected or any(not torch.isfinite(t).all() or t.shape!=model.state_dict()[n].shape for n,t in state.items()):raise ValueError('INCOMPLETE_READER')
        model.load_state_dict(state,strict=False);model.eval()
        results=[];routes={'base':[0.,0.,0.],'pro_fixed':[.5,.5,0.],'pro_matched':[.4,.4,0.],'reader':[0.,0.,1.],'max_fixed':[.4,.4,.2]}
        conditions=digest(dict(parent=sha256(a.parent_manifest),d=a.domain_sha,e=a.evidence_sha,r=a.reader_sha,dev=spec['sha256'],code=max_code(),greedy=True,max_new=768))
        for row in rr:
            for route,w in routes.items():
                z=generate(model,tok,row,w,'cuda',768);z.update(sample_id=row['sample_id'],fact_cluster_id=row['fact_cluster_id'],route=route,condition_sha=conditions,score=score(row,z['raw']))
                results.append(z)
                with (out/'RESPONSES.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(z,ensure_ascii=False)+'\n');f.flush()
        summary={route:dict(states=dict(collections.Counter(r['score'] for r in results if r['route']==route)),n=len(rr)) for route in routes}
        paired=[]
        for row in rr:
            values={r['route']:r['score']=='CORRECT' for r in results if r['sample_id']==row['sample_id']}
            paired.append(dict(sample_id=row['sample_id'],fact_cluster_id=row['fact_cluster_id'],**values))
        atomic(out/'REPORT.json',dict(summary=summary,paired=paired,independent_test=False,
              scope='REVEALED_SOURCE_EXTRACTION_ONLY_NOT_GENERAL_MEDICAL_CAPABILITY',Router_updates=0,Max_accepted=False,clinical_Gold=0,TEST2='SEALED'))


if __name__=='__main__':main()
