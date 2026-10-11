import argparse,json
from pathlib import Path
from PIL import Image
from .runtime import load,predict
from .assets import sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--manifest',default='configs/assets_manifest.json');p.add_argument('--image',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    g,r,s=load(a.assets,a.manifest,a.device)
    with Image.open(a.image) as im:z=predict(g,r,im,a.device)
    mask=z.pop('mask').numpy().astype('uint8')*255;Image.fromarray(mask).save(out/'mask.png')
    z.update(input_SHA256=sha(a.image),device=a.device,mask_resolution=[384,384],mask_projection='FULL_FRAME_STRETCH_TO_ORIGINAL',clinical_validation=False)
    (out/'prediction.json').write_text(json.dumps(z,ensure_ascii=False,indent=2),'utf8')
    print(json.dumps(z,ensure_ascii=False))
if __name__=='__main__':main()
