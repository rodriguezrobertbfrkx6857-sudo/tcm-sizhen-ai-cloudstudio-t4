"""Seeded synthetic visual inference; CUDA device selection requires HIP."""
import argparse,os,time
from pathlib import Path
try:from .amd_common import finish_report,require_hip,sha256
except ImportError:from amd_common import finish_report,require_hip,sha256
PIN='34ce40cd05168ff0cd637ee2e9d0dc7657dee1da47321be3bc37ad60492818b4'

def run(device='cpu',confirmed=False,model_path=None,pin=None):
    import torch
    r=dict(device=device,torch=torch.__version__,hip=torch.version.hip,training=False,downloads=False,seed=20261011,model_kind='synthetic_convnet')
    if device=='cuda':r['hardware']=require_hip(torch,confirmed);torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(r['seed']);torch.set_num_threads(2);start=time.perf_counter()
    net=torch.nn.Sequential(torch.nn.Conv2d(3,4,3,padding=1),torch.nn.ReLU(),torch.nn.AdaptiveAvgPool2d(1)).to(device).eval()
    with torch.no_grad():
        y=net(torch.linspace(0,1,3*64*64,device=device).reshape(1,3,64,64))
        if not torch.isfinite(y).all():raise ValueError('NONFINITE_VISUAL_OUTPUT')
        r['synthetic_mean']=float(y.mean());r['synthetic_output']=y.cpu().flatten().tolist()
    # Blank YOLO images may produce no boxes and skip NMS. Exercise its backend
    # explicitly with synthetic, non-medical boxes instead of assuming support.
    from torchvision.ops import nms
    boxes=torch.tensor([[0.,0.,100.,100.],[10.,10.,90.,90.],[200.,200.,300.,300.]],device=device)
    scores=torch.tensor([.9,.8,.7],device=device)
    selected=nms(boxes,scores,.5).cpu().tolist()
    if selected!=[0,2]:raise ValueError('NMS_BACKEND_CONTRACT_FAILURE')
    r['synthetic_nms_selected_indices']=selected
    if model_path:
        path=Path(model_path)
        # A user-supplied hash alone cannot establish trust of pickle code.
        if pin!=PIN or path.is_symlink() or sha256(path)!=PIN or path.stat().st_size!=5445914:raise ValueError('UNTRUSTED_YOLO_ASSET')
        os.environ['YOLO_AUTOINSTALL']='False'
        from ultralytics import YOLO
        import numpy as np
        m=YOLO(str(path))
        expected=['healthy_tongue','red_tongue','thin_tongue','white_coating','yellow_coating','black_coating']
        if m.task!='detect' or [m.names[i] for i in range(len(m.names))]!=expected:raise ValueError('MODEL_CLASS_CONTRACT_MISMATCH')
        result=m.predict(source=np.zeros((640,640,3),dtype=np.uint8),device=0 if device=='cuda' else 'cpu',imgsz=640,conf=.25,iou=.5,agnostic_nms=False,half=False,verbose=False,save=False)[0]
        r.update(model_kind='pinned_yolo_synthetic_only',checkpoint_SHA256=PIN,prediction_boxes=len(result.boxes),input_size=640,conf=.25,iou=.5,half=False,agnostic_nms=False,clinical_validation=False)
    if device=='cuda':torch.cuda.synchronize();r['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    r.update(status='PASS',seconds=time.perf_counter()-start)
    return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--confirm-no-other-compute',action='store_true');p.add_argument('--model');p.add_argument('--sha256');p.add_argument('--out');a=p.parse_args()
    try:r=run(a.device,a.confirm_no_other_compute,a.model,a.sha256)
    except Exception as exc:r=dict(status='FAIL',device=a.device,error_type=type(exc).__name__)
    finish_report(r,a.out);return 0 if r['status']=='PASS' else 2
if __name__=='__main__':raise SystemExit(main())
