"""Explicitly authorized local dataset evaluation, without dataset downloads."""
import argparse,json,os
from pathlib import Path
import yaml
from .yolo_infer import load_yolo
from .assets import sha

def validate_yaml(path,root):
    cfg=yaml.safe_load(Path(path).read_text('utf8'));root=Path(root).resolve()
    if not isinstance(cfg,dict) or 'download' in cfg:raise ValueError('NO_DATASET_DOWNLOAD_OR_CODE_ALLOWED')
    if any(x in str(path).lower() for x in ['test2','locked_test']):raise ValueError('SEALED_DATASET')
    base=Path(cfg.get('path','.'));base=(base if base.is_absolute() else root/base).resolve()
    if not base.is_relative_to(root):raise ValueError('DATASET_ROOT_ESCAPE')
    for key in ['train','val']:
        value=cfg.get(key)
        if not isinstance(value,str) or '://' in value:raise ValueError('LOCAL_SPLIT_DIRECTORY_REQUIRED')
        p=(base/value).resolve()
        if not p.is_relative_to(root) or not p.is_dir():raise ValueError('SPLIT_MISSING_OR_ESCAPE')
        if any(x in str(p).lower() for x in ['test2','locked_test']):raise ValueError('SEALED_DATASET')
    if cfg.get('test'):raise ValueError('NO_FINAL_TEST_IN_DEV_YAML')
    cfg['path']=str(base);return cfg

def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--manifest',default='configs/yolo_assets_manifest.json');p.add_argument('--data-root',required=True);p.add_argument('--data-yaml',required=True);p.add_argument('--confirm-authorized-data',action='store_true');p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--out',required=True);a=p.parse_args()
    if not a.confirm_authorized_data:raise ValueError('DATA_UPLOAD_AND_EVALUATION_PERMISSION_REQUIRED')
    cfg=validate_yaml(a.data_yaml,a.data_root);out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    settings=out/'ultralytics_settings';settings.mkdir();os.environ['YOLO_CONFIG_DIR']=str(settings)
    if a.device=='cuda':
        from .preflight import inspect
        inspect(require_gpu=True)
    model=load_yolo(a.assets,a.manifest)
    names=cfg.get('names');names={str(i):v for i,v in enumerate(names)} if isinstance(names,list) else {str(k):v for k,v in names.items()} if isinstance(names,dict) else {}
    if names!={str(k):v for k,v in model.names.items()}:raise ValueError('DATASET_CLASS_ORDER_MISMATCH')
    local=out/'AUTHORIZED_DEV_ONLY.yaml';local.write_text(yaml.safe_dump(cfg,allow_unicode=True),'utf8')
    result=model.val(data=str(local),split='val',imgsz=640,batch=1,device=0 if a.device=='cuda' else 'cpu',half=False,workers=0,project=str(out),name='metrics',exist_ok=False,plots=False,verbose=False)
    z=dict(status='ACTUAL_SOURCE_VAL_ONLY',metrics={k:float(v) for k,v in result.results_dict.items()},data_yaml_SHA256=sha(a.data_yaml),patient_independent_validation=False,clinical_validation=False,formal_training_updates=0)
    (out/'EVALUATION.json').write_text(json.dumps(z,indent=2),'utf8');print(json.dumps(z))
if __name__=='__main__':main()
