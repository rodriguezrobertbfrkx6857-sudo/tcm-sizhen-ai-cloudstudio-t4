"""Reuse existing DINO/ROI runtime with a HIP-specific resource guard."""
import argparse,json
from pathlib import Path
from scripts.amd_common import finish_report,require_hip,sha256
from scripts.asset_verify import verify

def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--manifest',default='configs/assets_manifest.json');p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--confirm-no-other-compute',action='store_true');p.add_argument('--out',required=True);a=p.parse_args()
    try:
        import torch
        from PIL import Image
        if a.device=='cuda':require_hip(torch,a.confirm_no_other_compute,min_free_gib=6)
        if verify(a.assets,a.manifest)['status']!='PASS':raise ValueError('LEGACY_ASSET_SHA_FAILURE')
        from cloudstudio_tongue.runtime import load,predict
        g,r,_=load(a.assets,a.manifest,a.device)
        result=predict(g,r,Image.new('RGB',(384,384),(127,127,127)),a.device)
        mask=result.pop('mask');result.update(status='PASS',device=a.device,mask_pixels=int(mask.sum()),mask_shape=list(mask.shape),input_kind='SYNTHETIC_SOLID_COLOR_NOT_PATIENT_DATA',clinical_validation=False)
    except Exception as exc:result=dict(status='FAIL',error_type=type(exc).__name__)
    finish_report(result,a.out);return 0 if result['status']=='PASS' else 2
if __name__=='__main__':raise SystemExit(main())
