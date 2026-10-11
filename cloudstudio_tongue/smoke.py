"""Bounded SYNTHETIC head smoke; never retrains or overwrites frozen weights."""
import argparse,json,time
from pathlib import Path
import torch
from .runtime import load
from .assets import sha
from .preflight import inspect

def run(root,manifest,out,device='cuda',steps=2):
    if not 1<=steps<=2:raise ValueError('SMOKE_MAX_TWO_STEPS')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    environment=inspect(require_gpu=device=='cuda');torch.manual_seed(17);torch.set_num_threads(2)
    g,r,s=load(root,manifest,device);g.eval();r.eval()
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    # Only frozen encoder features and classifier head; no clinical labels or data.
    x=torch.randn(1,3,224,224,device=device);start=time.perf_counter()
    amp=lambda:torch.autocast(device_type='cuda',dtype=torch.float16,enabled=device=='cuda')
    with torch.no_grad(),amp():
        feature=g.backbone(pixel_values=x,interpolate_pos_encoding=True).last_hidden_state[:,0,:].float()
        roi=r(torch.randn(1,3,384,384,device=device))
    assert torch.isfinite(roi).all()
    for p in g.classifier.parameters():p.requires_grad_(True)
    optimizer=torch.optim.SGD(g.classifier.parameters(),lr=1e-5);losses=[]
    for _ in range(steps):
        optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(g.classifier(feature),torch.zeros(1,dtype=torch.long,device=device));loss.backward()
        if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in g.classifier.parameters()) or not any(torch.count_nonzero(p.grad).item()>0 for p in g.classifier.parameters()):raise ValueError('BAD_LOSS_OR_GRADIENT')
        optimizer.step();losses.append(float(loss.detach()))
    before=g.classifier(feature).detach().cpu();checkpoint=out/'SYNTHETIC_HEAD_ONLY_NOT_RELEASE.pt'
    torch.save(dict(head={k:v.detach().cpu() for k,v in g.classifier.state_dict().items()},optimizer=optimizer.state_dict(),steps=steps,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else [],scope='SYNTHETIC_ENGINEERING_NOT_MODEL_RELEASE'),checkpoint)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True);head=torch.nn.Linear(384,2).to(device);head.load_state_dict(state['head'],strict=True)
    restored_optimizer=torch.optim.SGD(head.parameters(),lr=1e-5);restored_optimizer.load_state_dict(state['optimizer'])
    assert torch.equal(before,head(feature).detach().cpu())
    # Restore actual RNG state as well; this is not a formal training-resume proof.
    torch.set_rng_state(state['torch_rng'])
    if device=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng']);torch.cuda.synchronize()
    z=dict(status='SYNTHETIC_ENGINEERING_SMOKE_PASS',device=device,synthetic_head_updates=steps,formal_training_updates=0,loss=losses,elapsed_seconds=time.perf_counter()-start,checkpoint_SHA256=sha(checkpoint),head_reload_equal=True,environment=environment,
           CUDA_verified=device=='cuda',GPU_peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30 if device=='cuda' else None,clinical_metrics=None)
    (out/'SMOKE_REPORT.json').write_text(json.dumps(z,indent=2),'utf8');return z

def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--manifest',default='configs/assets_manifest.json');p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--steps',type=int,default=2);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(run(a.assets,a.manifest,a.out,a.device,a.steps)))
if __name__=='__main__':main()
