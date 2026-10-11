"""Build bounded reading supervision only from verified private TRAIN sources."""
import argparse
import json
import hashlib
from pathlib import Path
from qiora_rocm.training_data import rows,validate_source,isolate,digest,safe_path
from qiora_rocm.training_state import atomic


def target(sources):
    return json.dumps(dict(v='MAX_SOURCE_EXTRACT_V1',quotes=[dict(source_index=i,quote=s['evidence_text']) for i,s in enumerate(sources)]),ensure_ascii=False,separators=(',',':'))


def build(source_rows,role):
    selected={}
    for row in source_rows:
        for s in row.get('sources',[]):
            validate_source(s);key=digest([s['doc_id'],s['revision'],s['text_sha256']])
            selected.setdefault(key,dict(sample_id='MAX_READ_'+key,question='只从给定来源抽取原文。输出JSON：v为MAX_SOURCE_EXTRACT_V1，quotes是source_index与quote对象数组。没有来源时使用空数组，不补写原文。',
                answer=target([s]),sources=[s],fact_cluster_id=s.get('fact_cluster_id',row['fact_cluster_id']),
                work_id=s['work_id'],quality_tier='DETERMINISTIC_SOURCE_EXTRACTION_NOT_MEDICAL_GOLD',source_type=s['source_type'],
                train_role=role,split=role,role=role,medical_Gold=False,doctor_verified=False,loss_weight=1.))
    return list(selected.values())


def main():
    p=argparse.ArgumentParser();p.add_argument('--pro-data',required=True);p.add_argument('--manifest',default='configs/pro_revision03_data_manifest.json');p.add_argument('--out',required=True);a=p.parse_args()
    out=safe_path(a.out);out.mkdir(parents=True,exist_ok=False);m=json.loads(safe_path(a.manifest).read_text('utf8'));root=safe_path(a.pro_data)
    def load(k):
        s=m['datasets'][k];return rows(root/s['file'],s['sha256'],s['role'],s.get('row_selection'))
    train=build(load('E_COMPACT'),'TRAIN');dev=build(load('REVEALED_COMPACT_DEV'),'DEV')
    isolation=isolate(train,dev)
    if not train or not dev:raise ValueError('NO_REAL_READING_SOURCES')
    spec={}
    for name,rr in [('R_TRAIN',train),('R_DEV',dev)]:
        dest=out/(name+'.jsonl');dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr),encoding='utf8')
        spec[name]=dict(file=dest.name,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),rows=len(rr),role='TRAIN' if name=='R_TRAIN' else 'DEV')
    atomic(out/'MANIFEST.json',dict(version='MAX_READING_RESEARCH_V1',datasets=spec,isolation=isolation,medical_Gold=0,
         independent_test=False,source_manifest_sha=hashlib.sha256(safe_path(a.manifest).read_bytes()).hexdigest()))


if __name__=='__main__':main()
