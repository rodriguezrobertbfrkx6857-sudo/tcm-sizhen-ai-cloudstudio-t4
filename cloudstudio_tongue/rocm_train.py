"""Private six-class YOLO training and paired DEV on existing ROCm PyTorch."""
import argparse,json,os,shutil,time,uuid
from pathlib import Path
from .assets import sha
from .prepare_training import NAMES
from .private_state import safe_path,atomic,RunLock

def dataset(root,out):
    root=safe_path(root);m=json.loads((root/'DATA_MANIFEST.json').read_text('utf8'))
    if m['names']!=NAMES or not m['assets_copied']:raise ValueError('COPIED_PRIVATE_SIX_CLASS_DATA_REQUIRED')
    import yaml
    cfg=dict(path=str(root),names=NAMES)
    hashes={}
    for split,role in [('train','TRAIN'),('val','DEV')]:
        p=root/m['train_file' if role=='TRAIN' else 'dev_file'];key='train_sha' if role=='TRAIN' else 'dev_sha'
        if sha(p)!=m[key]:raise ValueError('ROW_MANIFEST_SHA_CHANGED')
        rr=[json.loads(x) for x in p.read_text('utf8').splitlines()];hashes[role]=set()
        if not rr:raise ValueError('EMPTY_SPLIT')
        for r in rr:
            for field in ('image','label'):
                asset=(root/r[field]).resolve()
                if not asset.is_relative_to(root) or sha(asset)!=r[field+'_sha']:raise ValueError('PRIVATE_ASSET_SHA_CHANGED_OR_ESCAPE')
            hashes[role].add(r['image_sha'])
        listing=out/(role+'_images.txt');listing.write_text(''.join(str(root/r['image'])+'\n' for r in rr),'utf8');cfg[split]=str(listing)
    if hashes['TRAIN']&hashes['DEV']:raise ValueError('EXACT_SPLIT_LEAKAGE')
    p=out/'PRIVATE_DEV_ONLY.yaml';p.write_text(yaml.safe_dump(cfg,allow_unicode=True),'utf8')
    return p,m

def backup_checkpoint(path,backup,epoch):
    path=Path(path)
    if not path.exists():raise ValueError('CHECKPOINT_SAVE_MISSING')
    dest=Path(backup)/('epoch_%04d_%s.pt'%(epoch,path.stem));h=sha(path)
    if dest.exists():
        if sha(dest)!=h:raise ValueError('BACKUP_COLLISION')
    else:
        tmp=dest.with_suffix('.partial');shutil.copyfile(path,tmp)
        with tmp.open('r+b') as f:f.flush();os.fsync(f.fileno())
        os.replace(tmp,dest)
    if sha(dest)!=h:raise ValueError('BACKUP_READBACK_FAILED')
    atomic(dest.with_suffix('.receipt.json'),dict(sha256=h,epoch=epoch,readback=True,instance_external_verified=False,full_reload='PENDING'))
    return dict(path=str(dest),sha256=h)

def main():
    started=time.time()
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['train','evaluate','restore'],required=True)
    p.add_argument('--weights',required=True);p.add_argument('--weights-sha',required=True);p.add_argument('--data-dir',required=True)
    p.add_argument('--out',required=True);p.add_argument('--backup-dir');p.add_argument('--baseline');p.add_argument('--baseline-sha')
    p.add_argument('--epochs',type=int,default=3);p.add_argument('--batch',type=int,default=8);p.add_argument('--max-seconds',type=int,default=3600)
    p.add_argument('--resume',action='store_true');p.add_argument('--confirm-private-research',action='store_true');a=p.parse_args()
    if not a.confirm_private_research:raise ValueError('PRIVATE_RESEARCH_CONFIRMATION_REQUIRED')
    if not 1<=a.epochs<=20 or not 1<=a.batch<=32 or not 0<a.max_seconds<=7200:raise ValueError('BOUNDED_PARAMETERS_REQUIRED')
    if sha(safe_path(a.weights))!=a.weights_sha:raise ValueError('TRUSTED_WEIGHT_SHA_REQUIRED_BEFORE_DESERIALIZATION')
    if a.mode=='restore':
        out=safe_path(a.out)
        if out.exists():raise ValueError('IMMUTABLE_RESTORE_PROOF_EXISTS')
        os.environ['YOLO_CONFIG_DIR']=str(out.parent/'vision_restore_settings')
        from ultralytics import YOLO
        import numpy as np
        model=YOLO(a.weights)
        if [model.names[i] for i in range(6)]!=NAMES:raise ValueError('RESTORE_CLASS_ORDER_CHANGED')
        result=model.predict(np.zeros((640,640,3),dtype=np.uint8),device='cpu',verbose=False)
        atomic(out,dict(status='ACTUAL_WEIGHT_CPU_RELOAD_AND_FORWARD_PASS',weight_sha=a.weights_sha,results=len(result),clinical_validation=False,optimizer_resume=False))
        return
    import torch
    if not torch.version.hip or not torch.cuda.is_available():raise ValueError('ROCM_REQUIRED')
    from torchvision.ops import nms
    if nms(torch.tensor([[0.,0.,1.,1.]],device='cuda'),torch.ones(1,device='cuda'),.5).numel()!=1:raise ValueError('ROCM_TORCHVISION_NMS_FAILED')
    out=safe_path(a.out)
    if out.exists() and not a.resume:raise ValueError('OUTPUT_EXISTS_EXPLICIT_RESUME_REQUIRED')
    if a.resume and a.mode!='train':raise ValueError('TRAIN_RESUME_ONLY')
    out.mkdir(parents=True,exist_ok=True);os.environ['YOLO_CONFIG_DIR']=str(out/'settings')
    from ultralytics import YOLO
    import ultralytics
    with RunLock(safe_path(a.weights).parent.parent/'QIORA_TRAINER.lock'):
        yaml_path,m=dataset(a.data_dir,out)
        model=YOLO(a.weights)
        if [model.names[i] for i in range(6)]!=NAMES:raise ValueError('MODEL_CLASS_ORDER_CHANGED')
        if a.mode=='evaluate':
            if not a.baseline or not a.baseline_sha or sha(safe_path(a.baseline))!=a.baseline_sha:raise ValueError('PAIRED_BASELINE_SHA_REQUIRED')
            metrics={}
            for name,obj in [('baseline',YOLO(a.baseline)),('candidate',model)]:
                r=obj.val(data=str(yaml_path),split='val',device=0,imgsz=640,batch=a.batch,half=False,workers=0,plots=False,project=str(out),name=name)
                metrics[name]={k:float(v) for k,v in r.results_dict.items()}
            atomic(out/'PAIRED_DEV.json',dict(metrics=metrics,delta={k:metrics['candidate'][k]-metrics['baseline'][k] for k in metrics['baseline']},patient_independent=False,clinical_validation=False,TEST2='NOT_READ'))
            return
        if not a.backup_dir:raise ValueError('BACKUP_REQUIRED')
        backup=safe_path(a.backup_dir)
        if backup==out or backup.is_relative_to(out) or out.is_relative_to(backup):raise ValueError('SEPARATE_BACKUP_REQUIRED')
        backup.mkdir(parents=True,exist_ok=True)
        plan=dict(data_manifest_sha=sha(Path(a.data_dir)/'DATA_MANIFEST.json'),epochs=a.epochs,batch=a.batch,max_seconds=a.max_seconds,imgsz=640,amp=False,torch=torch.__version__,hip=torch.version.hip,ultralytics=ultralytics.__version__,code_sha=sha(__file__))
        binding=out/'FROZEN_RUN.json'
        if a.resume:
            frozen=json.loads(binding.read_text('utf8'))
            if frozen['plan']!=plan:raise ValueError('RESUME_CONFIG_CHANGED')
        else:
            frozen=dict(plan=plan,start_unix=started,initial_weight_sha=a.weights_sha,nonce=uuid.uuid4().hex,resume_exact_bitwise=False)
            atomic(binding,frozen)
        deadline=frozen['start_unix']+a.max_seconds
        if time.time()>deadline-120:raise ValueError('ORIGINAL_WINDOW_EXHAUSTED')
        def save(trainer):
            receipts=[backup_checkpoint(trainer.last,backup,trainer.epoch)]
            if trainer.best.exists():receipts.append(backup_checkpoint(trainer.best,backup,trainer.epoch))
            atomic(out/'LATEST.json',dict(epoch=trainer.epoch,checkpoints=receipts,nonce=frozen['nonce']))
            shutil.copyfile(binding,backup/'FROZEN_RUN.json')
        def stop(trainer):
            if time.time()>deadline-120:trainer.stop=True
        def epoch_limit(trainer):
            if trainer.epoch+1>=a.epochs:trainer.stop=True
        model.add_callback('on_model_save',save);model.add_callback('on_train_batch_end',stop);model.add_callback('on_fit_epoch_end',epoch_limit)
        try:
            model.train(data=str(yaml_path),device=0,imgsz=640,batch=a.batch,epochs=a.epochs,workers=0,amp=False,optimizer='AdamW',lr0=1e-4,weight_decay=.0005,seed=20261013,deterministic=True,mosaic=0.,mixup=0.,hsv_h=0.,degrees=5.,translate=.05,scale=.1,save=True,save_period=1,plots=False,project=str(out),name='training',exist_ok=a.resume,resume=a.weights if a.resume else False,time=max(0,(deadline-time.time()-120)/3600))
            atomic(out/'STATUS.json',dict(status='VISION_RESEARCH_PENDING_PAIRED_DEV_AND_EXTERNAL_RELOAD',accepted=False,exact_resume=False,reason='Ultralytics native optimizer/EMA resume may round checkpoint tensors; not a bitwise continuation claim'))
        except BaseException as exc:
            atomic(out/'FAILURE.json',dict(error_type=type(exc).__name__,last_saved_only=True,nonce=frozen['nonce']));raise

if __name__=='__main__':main()
