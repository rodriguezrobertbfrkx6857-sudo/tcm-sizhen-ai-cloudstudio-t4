"""Raw-model paired DEV diagnostics; never reads sealed TEST2 or trains a Router."""
import argparse
import collections
import json
import os
import random
import re
import time
from pathlib import Path
import torch
from .training_data import rows, safe_path, prompt_ids, digest
from .training_state import atomic, read_owned, code_identity, RunLock, require_parent_binding
from .train import load_model

RELATIONS = {'SUPPORTED','PARTIAL','CONTRADICTED','IRRELEVANT','MISSING','WRONG_ATTRIBUTION'}


def relation(claim, sources):
    if not isinstance(claim,dict): return 'UNKNOWN'
    if not sources: return 'MISSING'
    if claim.get('kind') == 'modern_efficacy': return 'UNKNOWN'
    if claim.get('kind') == 'attribution':
        if not {'doc_id','work_id','revision'} <= claim.keys(): return 'UNKNOWN'
        return 'SUPPORTED' if any(s['doc_id']==claim['doc_id'] and s['work_id']==claim['work_id'] and str(s['revision'])==str(claim['revision']) for s in sources) else 'WRONG_ATTRIBUTION'
    atoms=claim.get('atoms',[])
    if not isinstance(atoms,list) or not atoms: return 'UNKNOWN'
    truth=[]
    for x in atoms:
        if not isinstance(x,dict) or not {'predicate','doc_id','text'}<=x.keys() or not isinstance(x['text'],str) or x['predicate'] not in ('text_contains','text_not_contains'): return 'UNKNOWN'
        doc=next((s for s in sources if s['doc_id']==x['doc_id']),None)
        if doc is None: truth.append(None); continue
        yes=x['text'] in doc['evidence_text']; truth.append(yes if x['predicate']=='text_contains' else not yes)
    if all(x is True for x in truth): return 'SUPPORTED'
    if any(x is False for x in truth): return 'CONTRADICTED'
    if any(x is True for x in truth): return 'PARTIAL'
    return 'IRRELEVANT'


def verdict(item, raw):
    if item.get('contract_version') == 'KER_COMPACT_RELATION_V3':
        try:
            a=json.loads(raw); s=item.get('sources',[])
            assert isinstance(a,dict) and set(a)=={'v','relation','evidence_ids'}
            assert a['v']=='KER_COMPACT_RELATION_V3' and a['relation'] in RELATIONS
            ids=a['evidence_ids']; assert isinstance(ids,list) and all(type(x) is int and 0<=x<len(s) for x in ids) and len(set(ids))==len(ids)
        except (ValueError,TypeError,AssertionError): return dict(state='NEEDS_REVIEW',category='INVALID_COMPACT_CONTRACT')
        expected=relation(item['claim'],s)
        if expected not in RELATIONS: return dict(state='NEEDS_REVIEW',category='UNKNOWN_SEMANTICS')
        ids=list(range(len(s))) if expected in {'SUPPORTED','PARTIAL','CONTRADICTED'} else []
        return dict(state='CORRECT' if a['relation']==expected and a['evidence_ids']==ids else 'INCORRECT',category='LITERAL_RELATION',relation=a['relation'])
    if item.get('kind') == 'EXACT_NONMEDICAL':
        return dict(state='CORRECT' if raw.strip()==item['expected_answer'] else 'INCORRECT',category='EXACT_GENERAL')
    if item.get('kind') in ('CHOICE','SINGLE_CHOICE','MULTI_CHOICE'):
        s=re.sub(r'^\s*答案\s*[:：]?','',raw).strip().strip('。.')
        if not re.fullmatch(r'[A-Z](?:[\s,，、]*[A-Z])*',s): return dict(state='NEEDS_REVIEW',category='UNPARSEABLE_CHOICE')
        letters=re.findall('[A-Z]',s)
        if len(set(letters))!=len(letters) or not all(x in item['options'] for x in letters): return dict(state='NEEDS_REVIEW',category='INVALID_CHOICE')
        return dict(state='CORRECT' if set(letters)==set(re.findall('[A-Z]',str(item['expected_answer']))) else 'INCORRECT',category='EXAM_WEAK_LABEL_NOT_DOCTOR_GOLD')
    # Intentionally no invented semantic/medical success from refusal keywords.
    return dict(state='NEEDS_REVIEW',category='NATURAL_LANGUAGE_OR_LEGACY_CONTRACT_REQUIRES_REVIEW')


def dimension(r):
    if r.get('contract_version')=='KER_COMPACT_RELATION_V3': return 'COMPACT_'+r['expected_relation']
    if r.get('kind')=='EXACT_NONMEDICAL': return 'GENERAL'
    if r.get('kind') in ('CHOICE','SINGLE_CHOICE','MULTI_CHOICE'): return 'KNOWLEDGE'
    return 'BOUNDARY_AND_LEGACY_MANUAL'


def paired_report(items, output):
    conditions={x['condition_sha'] for x in output}
    if len(conditions)!=1: raise ValueError('UNFAIR_CONDITIONS')
    routes=sorted({x['route'] for x in output}); by={(x['route'],x['sample_id']):x for x in output}
    if len(by)!=len(output) or any((route,r['sample_id']) not in by for route in routes for r in items): raise ValueError('INCOMPLETE_OR_DUPLICATE_PAIRS')
    result={}; details=[]; differences=[]
    for route in routes:
        vv=[]
        for r in items:
            o=by[route,r['sample_id']];v=verdict(r,o['raw']); vv.append((r,o,v))
            details.append(dict(o,verdict=v,dimension=dimension(r),fact_cluster_id=r['fact_cluster_id']))
        result[route]=dict(samples=len(vv),states=dict(collections.Counter(v['state'] for _,_,v in vv)),
                           truncated=sum(o['truncated'] for _,o,_ in vv))
        if route=='base': continue
        for dim in sorted({dimension(r) for r in items}):
            known=[]; failures=[]; new_errors=[]
            for r,o,v in vv:
                if dimension(r)!=dim: continue
                b=verdict(r,by['base',r['sample_id']]['raw'])
                if b['state'] in ('CORRECT','INCORRECT') and v['state'] in ('CORRECT','INCORRECT'):
                    known.append((r['fact_cluster_id'],int(b['state']=='CORRECT'),int(v['state']=='CORRECT')))
                if b['state']=='CORRECT' and v['state']=='INCORRECT': new_errors.append(r['sample_id'])
                if b!=v or o['raw']!=by['base',r['sample_id']]['raw']: failures.append(r['sample_id'])
            groups=collections.defaultdict(list)
            for cluster,b,c in known: groups[cluster].append(c-b)
            means=[sum(x)/len(x) for x in groups.values()];ci=None
            if means:
                rng=random.Random(20261011); draws=sorted(sum(rng.choices(means,k=len(means)))/len(means) for _ in range(2000))
                ci=[draws[49],draws[1949]]
            base=sum(b for _,b,_ in known)/len(known) if known else None
            candidate=sum(c for _,_,c in known)/len(known) if known else None
            result[route].setdefault('dimensions',{})[dim]=dict(paired_known_n=len(known),unknown_or_unpaired_n=sum(dimension(r)==dim for r in items)-len(known),
                fact_clusters=len(means),base_accuracy=base,candidate_accuracy=candidate,
                absolute_gain=None if base is None else candidate-base,relative_gain=None if not base else (candidate-base)/base,
                fact_cluster_macro_delta=None if not means else sum(means)/len(means),cluster_bootstrap_95_ci=ci,
                confirmed_new_programmatic_errors=new_errors,changed_output_ids=failures)
            differences += [dict(route=route,dimension=dim,sample_id=x) for x in new_errors]
    return dict(protocol='AMD_DEV_DIAGNOSTIC_V1_NOT_HISTORICAL_SCORER_REPLACEMENT',summary=result,details=details,
        confirmed_new_errors=differences,Gate='BLOCK_CRITICAL_REGRESSION_OR_MANUAL_REVIEW_REQUIRED',
        PRO_V1_INITIAL_ACCEPTED=False,independent_test=False,medical_Gold=0,TEST2='SEALED',
        citation_skill='SYSTEM_SOURCE_BINDING_NOT_MODEL_GENERATED_CITATION',router_updates=0)


def fuse(model, domain=None, evidence=None):
    state=model.state_dict()
    for expert, cp in [('domain',domain),('evidence',evidence)]:
        if cp is None: continue
        if cp.get('config')!=model.pro_config.to_dict() or cp.get('phase')!=('T1' if expert=='domain' else 'T2'):
            raise ValueError('EXPERT_CHECKPOINT_CONFIG_OR_PHASE_MISMATCH')
        names={n for n in state if f'lora_A.{expert}.' in n or f'lora_B.{expert}.' in n}
        chosen={n:t for n,t in cp['adapter_router_state'].items() if n in names}
        if set(chosen)!=names or any(t.shape!=state[n].shape or t.dtype!=state[n].dtype or not torch.isfinite(t).all() for n,t in chosen.items()):
            raise ValueError('INCOMPLETE_EXPERT_WEIGHTS')
        model.load_state_dict(chosen,strict=False)
    model.set_phase('INFERENCE');model.eval()


def generate(model, tok, item, weights, device, max_new):
    ids=prompt_ids(tok,item)
    if len(ids)+max_new>model.backbone.config.max_position_embeddings: raise ValueError('EVAL_CONTEXT_OVERFLOW')
    x=torch.tensor([ids],device=device);mask=torch.ones_like(x)
    with torch.inference_mode(), model.routed(torch.tensor([weights],device=device)):
        y=model.backbone.generate(input_ids=x,attention_mask=mask,do_sample=False,num_beams=1,
            max_new_tokens=max_new,use_cache=True,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
    suffix=y[0,len(ids):].tolist()
    return dict(raw=tok.decode(suffix,skip_special_tokens=True),generated_tokens=len(suffix),
                truncated=len(suffix)==max_new and (not suffix or suffix[-1]!=tok.eos_token_id))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',required=True);p.add_argument('--parent-manifest',default='configs/qiora_parent_assets_manifest.json')
    p.add_argument('--data-dir',required=True);p.add_argument('--data-manifest',default='configs/pro_revision03_data_manifest.json')
    p.add_argument('--domain');p.add_argument('--domain-sha');p.add_argument('--evidence');p.add_argument('--evidence-sha')
    p.add_argument('--out',required=True);p.add_argument('--resume-eval',action='store_true');a=p.parse_args()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    out=safe_path(a.out)
    if out.exists() and not a.resume_eval:raise ValueError('EXPLICIT_EVAL_RESUME_REQUIRED')
    if a.resume_eval and not out.exists():raise ValueError('NO_EVAL_TO_RESUME')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'REPORT.json').exists():raise ValueError('COMPLETED_REPORT_IMMUTABLE')
    mf=json.loads(safe_path(a.data_manifest).read_text('utf8'));items=[]
    for name in mf['dev_sets']:
        spec=mf['datasets'][name];items+=rows(safe_path(a.data_dir)/spec['file'],spec['sha256'],'DEV')
    dc=read_owned(a.domain,a.domain_sha) if a.domain else None;ec=read_owned(a.evidence,a.evidence_sha) if a.evidence else None
    for cp in (dc,ec):
        if cp is not None:require_parent_binding(cp,sha256(a.parent_manifest))
    if len({r['sample_id'] for r in items})!=len(items):raise ValueError('CROSS_PANEL_DUPLICATE_IDS')
    with RunLock(safe_path(a.parent).parent/'QIORA_TRAINER.lock'):
        model,tok,parent=load_model(safe_path(a.parent),a.parent_manifest)
        fuse(model,dc,ec);responses=[]
        routes={'base':[0.,0.,0.]}
        if dc:routes['domain']=[1.,0.,0.]
        if ec:routes['evidence']=[0.,1.,0.]
        if dc and ec:routes.update(fixed=[.5,.5,0.],rule=None)
        conditions=dict(parent_sha=sha256(a.parent_manifest),dev_manifest_sha=sha256(a.data_manifest),code_sha=code_identity(),
                        experts=dict(domain=a.domain_sha,evidence=a.evidence_sha),max_new_legacy=512,max_new_compact=192,
                        do_sample=False,num_beams=1,precision='BF16',rule_is_coefficient_selection_not_output_replacement=True)
        h=digest(conditions)
        if a.resume_eval:
            if json.loads((out/'CONDITIONS.json').read_text('utf8'))!=conditions:raise ValueError('EVAL_RESUME_CONDITIONS_CHANGED')
            responses=[json.loads(x) for x in (out/'RESPONSES.jsonl').read_text('utf8').splitlines()] if (out/'RESPONSES.jsonl').exists() else []
            if any(r['condition_sha']!=h for r in responses):raise ValueError('EVAL_CACHE_CONDITIONS_CHANGED')
        else:atomic(out/'CONDITIONS.json',conditions)
        done={(r['sample_id'],r['route']) for r in responses}
        allowed={(r['sample_id'],route) for r in items for route in routes}
        if len(done)!=len(responses) or not done<=allowed:raise ValueError('INVALID_EVAL_PARTIAL_OUTPUTS')
        for item in items:
            for route,w in routes.items():
                if (item['sample_id'],route) in done:continue
                if route=='rule':w=[.2,.8,0.] if item.get('sources') else [.8,.2,0.]
                t=time.perf_counter();o=generate(model,tok,item,w,'cuda',192 if item.get('contract_version')=='KER_COMPACT_RELATION_V3' else 512)
                o.update(sample_id=item['sample_id'],route=route,condition_sha=h,seconds=time.perf_counter()-t,coefficients=w)
                responses.append(o)
                with (out/'RESPONSES.jsonl').open('a',encoding='utf8') as f:
                    f.write(json.dumps(o,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
        atomic(out/'REPORT.json',paired_report(items,responses))


if __name__=='__main__':main()
