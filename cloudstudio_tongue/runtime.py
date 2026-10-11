"""Load only explicitly staged, fingerprinted visual weights."""
import json
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageOps
from .assets import verify
from .models import GlobalClassifier,ResNet18UNet,Dinov2Config,Dinov2Model

def load(root,manifest,device='cpu'):
    spec,paths=verify(root,manifest)
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA_UNAVAILABLE')
    cfg=Dinov2Config.from_json_file(str(paths['dino_config']))
    if (cfg.hidden_size,cfg.num_hidden_layers,cfg.patch_size)!=(384,12,14):raise ValueError('DINO_ARCHITECTURE_MISMATCH')
    # Both checkpoints contain the FULL model state, including encoder tensors.
    # The base file is SHA-verified for lineage, never merged or retrained.
    g=GlobalClassifier(Dinov2Model(cfg));r=ResNet18UNet()
    gs=torch.load(paths['g1_checkpoint'],map_location='cpu',weights_only=True)
    rs=torch.load(paths['roi_checkpoint'],map_location='cpu',weights_only=True)
    for k,v in spec['g1_checkpoint_metadata'].items():
        if gs.get(k)!=v:raise ValueError('G1_METADATA_MISMATCH:'+k)
    if rs.get('schema_version')!='ROI_R1_CHECKPOINT_V1' or rs.get('model_type')!='resnet18_unet':raise ValueError('ROI_METADATA_MISMATCH')
    if rs.get('training_code_sha256')!=spec['roi_training_code_sha256']:raise ValueError('ROI_CODE_LINEAGE_MISMATCH')
    for model,state in [(g,gs['model_state_dict']),(r,rs['model_state_dict'])]:
        model.load_state_dict(state,strict=True)
        if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('NONFINITE_WEIGHTS')
        model.eval().to(device)
        for p in model.parameters():p.requires_grad_(False)
    return g,r,spec

def pixels(image,size,device):
    image=ImageOps.exif_transpose(image).convert('RGB').resize((size,size),Image.Resampling.BICUBIC if size==224 else Image.Resampling.BILINEAR)
    x=torch.from_numpy(np.asarray(image,dtype=np.float32).copy()/255).permute(2,0,1).unsqueeze(0)
    mean=torch.tensor([.485,.456,.406]).view(1,3,1,1);std=torch.tensor([.229,.224,.225]).view(1,3,1,1)
    return ((x-mean)/std).to(device)

def predict(g,r,image,device='cpu'):
    with torch.inference_mode():
        # FP32 default preserves the CPU baseline; FP16 tested separately in smoke.
        p=g(pixels(image,224,device)).softmax(-1)[0].cpu().tolist()
        mask=r(pixels(image,384,device)).sigmoid()[0,0]>=.5
    return dict(class_order=['white_coating','yellow_coating'],probabilities=p,mask=mask.cpu(),
                scope='SOURCE_LIMITED_RESEARCH_COMPONENT_NOT_DIAGNOSIS',pipeline='G1_FULL_FRAME_AND_ROI_SEPARATE_NO_ROI_CROP_FEEDBACK')
