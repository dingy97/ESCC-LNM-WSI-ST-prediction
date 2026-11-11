"""Frozen encoders on every existing physical patch; no RNA or clinical label input."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
from bridge_core import pool_features
from image_qc import relative_od_fraction

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'03_bridge_training/features'
OLD=Path(r'__ESCC_LEGACY_ROOT__')

def main():
    torch.set_num_threads(2);torch.manual_seed(20261104);np.random.seed(20261104)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    table=pd.read_csv(ROOT/'fov_patch_manifest.csv')
    assert len(table)==6176 and table.patch_id.is_unique
    from torchvision.models import resnet50
    weight=Path.home()/'.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth'
    digest=hashlib.sha256(weight.read_bytes()).hexdigest();assert digest.startswith('11ad3fa6')
    sys.path.insert(0,str(OLD/'runtime_deps'))
    encoders=['ResNet50','Phikon']
    maskrows=[]
    for name in encoders:
        file=OUT/f'{name}_patch_features.npz'
        if file.exists():
            a=np.load(file);assert np.array_equal(a['patch_id'],table.patch_id.to_numpy());features=a['features']
            print(f'Reusing exact-key {name} features',flush=True)
        else:
            if name=='ResNet50':
                model=resnet50(weights=None)
                model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True))
                model.fc=torch.nn.Identity();model=model.train(False).to(device)
            else:
                from transformers import ViTModel
                model=ViTModel.from_pretrained(OLD/'weights/phikon',add_pooling_layer=False,local_files_only=True).train(False).to(device)
            for parameter in model.parameters():parameter.requires_grad_(False)
            started=time.monotonic();batches=[]
            for begin in range(0,len(table),24):
                items=table.iloc[begin:begin+24]
                images=[]
                for row in items.itertuples():
                    im=np.asarray(Image.open(row.patch_path).convert('RGB'))
                    assert im.shape==(224,224,3)
                    images.append(im)
                    if name=='ResNet50':
                        fraction=relative_od_fraction(im)
                        maskrows.append({'patch_id':row.patch_id,'FOV_ID':row.FOV_ID,
                             'OD_tissue_fraction':fraction,'HSV_kept':row.eligible_for_encoder})
                x=torch.from_numpy(np.stack(images)).permute(0,3,1,2).to(device,dtype=torch.float32)/255.
                mu=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
                sd=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
                x=(x-mu)/sd
                with torch.inference_mode():
                    result=model(x) if name=='ResNet50' else model(pixel_values=x).last_hidden_state[:,0]
                    batches.append(result.cpu().numpy().astype('float32'))
                if begin%480==0:print(f'{name}: {min(begin+24,len(table))}/{len(table)}, {time.monotonic()-started:.0f}s',flush=True)
            features=np.concatenate(batches)
            assert np.isfinite(features).all()
            np.savez_compressed(file,features=features,patch_id=table.patch_id.to_numpy(dtype=str))
            del model;torch.cuda.empty_cache()
        if name=='ResNet50' and maskrows:
            pd.DataFrame(maskrows).to_csv(OUT/'patch_OD_mask_audit.csv',index=False)
        masks=pd.read_csv(OUT/'patch_OD_mask_audit.csv').set_index('patch_id').reindex(table.patch_id)
        for aggregation in ['HSV_mean','all16_mean','OD_soft_mean']:
            ids=[];vectors=[]
            for fid,indices in table.groupby('FOV_ID').indices.items():
                if aggregation=='HSV_mean':weights=table.iloc[indices].eligible_for_encoder.to_numpy(dtype=float)
                elif aggregation=='all16_mean':weights=np.ones(len(indices))
                else:weights=np.maximum(masks.iloc[indices].OD_tissue_fraction.to_numpy(),.02)
                # FOV386 has no hard-mask tissue. Keep its key and NA embedding; never train it.
                vector=pool_features(features[indices],weights) if weights.sum()>0 else np.full(features.shape[1],np.nan)
                ids.append(fid);vectors.append(vector)
            np.savez_compressed(OUT/f'{name}_{aggregation}_FOV_features.npz',FOV_ID=np.array(ids),features=np.asarray(vectors,dtype='float32'))
        metadata={'encoder':name,'frozen':True,'patches':len(table),'dimension':features.shape[1],
          'precision':'FP32','device':device,'seed':20261104,'native_patch_input':'existing 224px, approximately 0.5um/px',
          'resize_or_thumbnail_input':False,'normalization':'ImageNet RGB mean/std',
          'weight_path':str(weight if name=='ResNet50' else OLD/'weights/phikon/model.safetensors'),
          'weight_sha256':digest if name=='ResNet50' else hashlib.sha256((OLD/'weights/phikon/model.safetensors').read_bytes()).hexdigest(),
          'pretraining_exposure':'ImageNet' if name=='ResNet50' else 'TCGA, explicitly disclosed for later TCGA evaluation',
          'RNA_or_LNM_used_in_extraction':False}
        (OUT/f'{name}_provenance.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('All frozen patch and FOV features completed',flush=True)

if __name__=='__main__':main()
