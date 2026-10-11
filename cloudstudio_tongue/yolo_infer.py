"""Current six-class appearance model; only the pinned trusted checkpoint loads."""
import argparse,json,os
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw
from .assets import verify
PIN='34ce40cd05168ff0cd637ee2e9d0dc7657dee1da47321be3bc37ad60492818b4'
LABELS=['健康舌','红舌','瘦舌','白苔舌','黄苔舌','黑苔舌']

def load_yolo(root,manifest):
    # YOLO serialization contains Python module objects. SHA verification of this
    # known checkpoint must precede Ultralytics' trusted deserialization.
    spec,paths=verify(root,manifest)
    entry=next(x for x in spec['assets'] if x['id']=='yolo_six_class')
    if entry['sha256']!=PIN:raise ValueError('NOT_PINNED_TRUSTED_YOLO_CHECKPOINT')
    from ultralytics import YOLO
    model=YOLO(str(paths['yolo_six_class']))
    if model.task!='detect' or len(model.names)!=6:raise ValueError('YOLO_TASK_OR_CLASS_COUNT_MISMATCH')
    expected=spec['model_class_names']
    if {str(k):v for k,v in model.names.items()}!=expected:raise ValueError('YOLO_CLASS_MAPPING_MISMATCH')
    return model

def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--manifest',default='configs/yolo_assets_manifest.json');p.add_argument('--image',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);os.environ['YOLO_CONFIG_DIR']=str(out/'ultralytics_settings')
    (out/'ultralytics_settings').mkdir()
    if a.device=='cuda':
        from .preflight import inspect
        inspect(require_gpu=True)
    m=load_yolo(a.assets,a.manifest)
    with Image.open(a.image) as source:
        if source.width*source.height>24_000_000:raise ValueError('IMAGE_TOO_LARGE')
        image=ImageOps.exif_transpose(source).convert('RGB')
        result=m.predict(image,imgsz=640,device=0 if a.device=='cuda' else 'cpu',conf=.25,iou=.5,agnostic_nms=False,half=False,verbose=False)[0]
    boxes=[dict(class_id=int(b.cls.item()),label=LABELS[int(b.cls.item())],confidence=float(b.conf.item()),xyxy=b.xyxy[0].cpu().tolist()) for b in result.boxes]
    z=dict(model='YOLO11n_SIX_CLASS_SOURCE_LIMITED',checkpoint_SHA256=PIN,device=a.device,detections=boxes,clinical_validation=False,patient_independent_validation=False,conf=.25,iou=.5,input_size=640)
    (out/'prediction.json').write_text(json.dumps(z,ensure_ascii=False,indent=2),'utf8')
    # Pure Pillow drawing avoids Ultralytics downloading fonts for annotations.
    overlay=image.copy();draw=ImageDraw.Draw(overlay)
    for b in boxes:
        draw.rectangle(b['xyxy'],outline='red',width=2);draw.text((b['xyxy'][0],b['xyxy'][1]),str(b['class_id']),fill='red')
    overlay.save(out/'overlay.png');print(json.dumps(z,ensure_ascii=False))
if __name__=='__main__':main()
