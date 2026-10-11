"""Optional authorized DEV component evaluation; no data included in this repo."""
import argparse,json
from pathlib import Path
from PIL import Image
import numpy as np
from .runtime import load,predict
from .assets import sha

def safe(root,value):
    path=Path(value)
    if path.is_absolute() or '..' in path.parts:raise ValueError('DATA_PATH_ESCAPE')
    p=(root/path).resolve()
    if not p.is_relative_to(root):raise ValueError('DATA_PATH_ESCAPE')
    return p

def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--manifest',default='configs/assets_manifest.json');p.add_argument('--data-root',required=True);p.add_argument('--dev-manifest',required=True);p.add_argument('--confirm-authorized-data',action='store_true');p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--out',required=True);a=p.parse_args()
    if not a.confirm_authorized_data:raise ValueError('DATA_USE_NOT_CONFIRMED')
    if 'test2' in a.dev_manifest.lower():raise ValueError('SEALED_TEST_NOT_ALLOWED')
    rr=[json.loads(x) for x in Path(a.dev_manifest).read_text('utf8').splitlines() if x.strip()]
    if not rr or len({x['id'] for x in rr})!=len(rr):raise ValueError('EMPTY_OR_DUPLICATE_CASES')
    if any(x.get('split')!='DEV' or x.get('use_authorized') is not True for x in rr):raise ValueError('DEV_OR_USE_SCOPE_MISMATCH')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);root=Path(a.data_root).resolve();g,r,s=load(a.assets,a.manifest,a.device);dice=[];confusion=[[0,0],[0,0]];unknown=0
    for row in rr:
        with Image.open(safe(root,row['image'])) as im:z=predict(g,r,im,a.device)
        label=row.get('coating_label')
        if label in ['white_coating','yellow_coating']:
            y=['white_coating','yellow_coating'].index(label);guess=int(np.argmax(z['probabilities']));confusion[y][guess]+=1
        else:unknown+=1
        if row.get('mask'):
            with Image.open(safe(root,row['mask'])) as im:truth=np.asarray(im.convert('L').resize((384,384),Image.Resampling.NEAREST))>=128
            mask=z['mask'].numpy();dice.append(float((2*(mask&truth).sum()+1)/(mask.sum()+truth.sum()+1)))
    n=sum(map(sum,confusion));report=dict(status='ACTUAL_COMPONENT_DEV_ONLY',cases=len(rr),known_classification_cases=n,unknown_labels_excluded=unknown,confusion_matrix=confusion,conditional_accuracy=(confusion[0][0]+confusion[1][1])/n if n else None,segmentation_cases=len(dice),mean_mask_dice=float(np.mean(dice)) if dice else None,DEV_manifest_SHA256=sha(a.dev_manifest),independent_clinical_validation=False,claims='Publisher-label class DEV / given-mask geometry only')
    (out/'EVALUATION.json').write_text(json.dumps(report,indent=2),'utf8');print(json.dumps(report))
if __name__=='__main__':main()
