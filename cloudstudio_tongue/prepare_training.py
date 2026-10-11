"""Private six-class dataset, exact/near isolation and prior DEV/probe exclusion."""
import argparse
import collections
import hashlib
import json
import math
import shutil
from pathlib import Path
from PIL import Image,ImageOps
from .private_state import safe_path,atomic
from .assets import sha

NAMES=['healthy_tongue','red_tongue','thin_tongue','white_coating','yellow_coating','black_coating']


def image_row(root,p,split):
    label=root/'labels'/split/(p.stem+'.txt');boxes=[]
    for line in label.read_text('utf8').splitlines():
        if not line.strip():continue
        x=[float(v) for v in line.split()]
        if len(x)!=5 or not all(math.isfinite(v) for v in x):raise ValueError('INVALID_BOX')
        c,cx,cy,w,h=x
        if c!=int(c) or not 0<=c<6 or not 0<w<=1 or not 0<h<=1 or cx-w/2<-.00001 or cy-h/2<-.00001 or cx+w/2>1.00001 or cy+h/2>1.00001:raise ValueError('BOX_OUT_OF_RANGE')
        boxes.append(x)
    if not boxes:raise ValueError('NO_ANNOTATED_BOX_UNKNOWN_NOT_NEGATIVE')
    with Image.open(p) as im:
        im.load();im=ImageOps.exif_transpose(im).convert('RGB');pixels=list(im.resize((9,8)).convert('L').getdata())
        dh=sum((pixels[y*9+x]>pixels[y*9+x+1])<<(y*8+x) for y in range(8) for x in range(8))
    return dict(image=p.relative_to(root).as_posix(),label=label.relative_to(root).as_posix(),image_sha=sha(p),label_sha=sha(label),dhash=dh,split=split,boxes=boxes)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--known-dev-manifest',required=True);p.add_argument('--probe-signatures',nargs=2,required=True)
    p.add_argument('--out',required=True);p.add_argument('--copy-assets',action='store_true');a=p.parse_args()
    root=safe_path(a.source);out=safe_path(a.out)
    if out==root or out.is_relative_to(root) or root.is_relative_to(out):raise ValueError('PRIVATE_OUTPUT_MUST_BE_SEPARATE')
    out.mkdir(parents=True,exist_ok=False)
    names=[x.strip() for x in (root/'class.txt').read_text('utf-8-sig').splitlines() if x.strip()]
    if names!=['健康舌','红舌','瘦舌','白苔舌','黄苔舌','黑苔舌']:raise ValueError('CLASS_MAPPING_MISMATCH')
    frozen={r['image_sha256'] for r in [json.loads(x) for x in safe_path(a.known_dev_manifest).read_text('utf8').splitlines()] if r['split']!='TRAIN'}
    probes=[]
    for path in a.probe_signatures:
        probes += [(r['sha256'],int(r['dhash'],16)) for r in [json.loads(x) for x in safe_path(path).read_text('utf8').splitlines()]]
    allrows=[];errors=[]
    for split in ('train','val'):
        for f in sorted((root/'images'/split).iterdir()):
            if not f.is_file():continue
            try:allrows.append(image_row(root,f,split))
            except (ValueError,OSError) as exc:errors.append(dict(path=f.relative_to(root).as_posix(),reason=str(exc)))
    dev=[r for r in allrows if r['split']=='val'];dh={r['image_sha'] for r in dev};ph={h for h,_ in probes};seen=set();train=[];excluded=collections.Counter()
    for r in allrows:
        if r['split']!='train':continue
        reason=None
        if r['image_sha'] in ph or any((r['dhash']^d).bit_count()<=4 for _,d in probes):reason='PROTECTED_EXTERNAL_PROBE'
        elif r['image_sha'] in frozen:reason='KNOWN_FROZEN_DEV'
        elif r['image_sha'] in dh:reason='EXACT_TRAIN_VAL_OVERLAP'
        elif r['image_sha'] in seen:reason='EXACT_TRAIN_DUPLICATE'
        elif any((r['dhash']^v['dhash']).bit_count()<=4 for v in dev):reason='NEAR_TRAIN_VAL_DHASH4'
        if reason:excluded[reason]+=1
        else:train.append(r);seen.add(r['image_sha'])
    hard=[e for e in errors if 'truncated' not in e['reason']]
    if hard or len(errors)>2 or not train or not dev:
        atomic(out/'BLOCKED.json',dict(errors=errors));raise ValueError('DATA_ADMISSION_FAILURE')
    for name,rr in [('TRAIN',train),('DEV',dev)]:
        (out/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr),encoding='utf8')
        if a.copy_assets:
            for r in rr:
                for k in ('image','label'):
                    target=out/r[k];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/r[k],target)
                    if sha(target)!=r[k+'_sha']:raise ValueError('ASSET_COPY_SHA_MISMATCH')
    atomic(out/'DATA_MANIFEST.json',dict(format='SIX_CLASS_APPEARANCE_DEV_RESEARCH_V1',names=NAMES,
          train_file='TRAIN.jsonl',dev_file='DEV.jsonl',train_sha=sha(out/'TRAIN.jsonl'),dev_sha=sha(out/'DEV.jsonl'),
          train_rows=len(train),dev_rows=len(dev),exclusions=dict(excluded),quarantined=errors,assets_copied=a.copy_assets,
          source_signatures={str(i)+'_'+Path(x).name:sha(x) for i,x in enumerate([a.known_dev_manifest,*a.probe_signatures])},clinical_Gold=False,patient_independent=False,TEST2='NOT_READ'))


if __name__=='__main__':main()
