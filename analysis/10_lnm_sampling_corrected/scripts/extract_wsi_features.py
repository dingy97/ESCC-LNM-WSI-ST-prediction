"""All local ESD WSIs; frozen physical-FOV encoders and frozen RNA bridge."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
from common import ROOT, OUT, SEED, digest, write_json
from pathlib import Path
import io
import json
import sys
import time
import cv2
import joblib
import numpy as np
import openslide
import pandas as pd
from PIL import Image
import torch
from spatial_sampling import tissue_mask, slide_candidate_mask, candidate_fovs, patch_starts
from broad_inference import BroadBridgeBundle

OLD=Path(r'__ESCC_LEGACY_ROOT__')

def record_exclusion(row,error):
    directory=OUT/'inventory/excluded_WSI';directory.mkdir(exist_ok=True)
    record={'slide_id':row.slide_id,'Patient_ID':row.Patient_ID,'source_path':row.path,
        'reason':str(error),'status':'excluded_image_read_or_tissue_QC_failure',
        'authorization':'User 2026-11-05: 报错的就跳过','original_modified':False}
    write_json(directory/f'{row.slide_id}.json',record)
    print(f'SKIPPED {row.slide_id}: {error}',flush=True)
    return record

def load_encoders():
    from torchvision.models import resnet50
    sys.path.insert(0,str(OLD/'runtime_deps'))
    from transformers import ViTModel
    res=resnet50(weights=None)
    weights=Path.home()/'.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth'
    assert digest(weights)=='11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca'
    res.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
    res.fc=torch.nn.Identity()
    phi=ViTModel.from_pretrained(OLD/'weights/phikon',add_pooling_layer=False,local_files_only=True)
    for model in (phi,res):
        model.train(False).to('cuda')
        for param in model.parameters(): param.requires_grad_(False)
    return phi,res

def encode(images, models):
    results=[[],[]]
    for start in range(0,len(images),24):
        x=torch.from_numpy(np.stack(images[start:start+24])).permute(0,3,1,2).to('cuda',dtype=torch.float32)/255
        mu=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
        sd=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
        x=(x-mu)/sd
        with torch.inference_mode():
            results[0].append(models[0](pixel_values=x).last_hidden_state[:,0].cpu().numpy())
            results[1].append(models[1](x).cpu().numpy())
    return [np.concatenate(r) for r in results]

def read_patches(slide, coords, mpp, fov_um):
    images=[]; groups=[]; fractions=[]; perbag=[]
    fw,fh=np.asarray(fov_um)/np.asarray(mpp)
    for bag,(x,y) in enumerate(coords):
        # Read Level 0, resample one complete physical FOV to the Stage-2 1024-pixel grid.
        region=np.asarray(slide.read_region((int(x),int(y)),0,(round(fw),round(fh))).convert('RGB'))
        fov=cv2.resize(region,(1024,1024),interpolation=cv2.INTER_LINEAR)
        row=[]
        for py in patch_starts(1024):
            for px in patch_starts(1024):
                patch=fov[py:py+224,px:px+224]
                # Match the Stage-2 derived JPEG codec before the encoder.
                buffer=io.BytesIO(); Image.fromarray(patch).save(buffer,format='JPEG',quality=95,subsampling=0)
                decoded=np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert('RGB'))
                fraction=float(tissue_mask(decoded).mean()); row.append(fraction)
                if fraction>=.05:
                    images.append(decoded); groups.append(bag); fractions.append(fraction)
        perbag.append(row)
    return images,np.asarray(groups),np.asarray(perbag)

def main():
    config=json.loads((OUT/'config/frozen_analysis.json').read_text(encoding='utf-8'))
    config_hash=digest(OUT/'config/frozen_analysis.json')
    torch.set_num_threads(2); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); np.random.seed(SEED)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True; torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available()
    models=load_encoders()
    bridge=BroadBridgeBundle(ROOT/'03_bridge_training/bridge_model_checkpoints/broad_transcriptome')
    auxiliary=joblib.load(ROOT/'03_bridge_training/bridge_model_checkpoints/exploratory_full_embedding_Ridge/model.joblib')
    manifest=pd.read_csv(OUT/'inventory/WSI_manifest.csv').sort_values('slide_id')
    metadata=pd.read_csv(OUT/'inventory/WSI_metadata.csv',dtype={'mtime_ns':str}).set_index('slide_id')
    started=time.monotonic();successful=[];excluded=[]
    for number,row in enumerate(manifest.itertuples(),1):
        destination=OUT/f'features/slides/{row.slide_id}.npz'
        meta=destination.with_suffix('.json')
        stat=Path(row.path).stat(); original=metadata.loc[row.slide_id]
        assert stat.st_size==original['bytes'] and str(stat.st_mtime_ns)==original.mtime_ns
        if destination.exists() and meta.exists():
            old=json.loads(meta.read_text(encoding='utf-8'))
            assert old['config_sha256']==config_hash
            successful.append(row.slide_id)
            print(f'{number}/651 CACHED {row.slide_id}',flush=True); continue
        began=time.monotonic()
        try:
            with openslide.OpenSlide(row.path) as slide:
                mpp=[float(slide.properties[openslide.PROPERTY_NAME_MPP_X]),float(slide.properties[openslide.PROPERTY_NAME_MPP_Y])]
                thumb=slide.get_thumbnail((2048,2048)).convert('RGB')
                thumb.save(OUT/f'features/thumbnails/{row.slide_id}.jpg',quality=90)
                candidate_mask,mask_qc=slide_candidate_mask(np.asarray(thumb),slide.dimensions,mpp)
                coords,qc=candidate_fovs(slide.dimensions,mpp,candidate_mask,config['FOV_um'],config['max_virtual_FOVs_per_slide'])
                qc.update(mask_qc)
                if len(coords)==0:
                    excluded.append(record_exclusion(row,'No tissue FOV candidates'));continue
                images,groups,fractions=read_patches(slide,coords,mpp,config['FOV_um'])
        except openslide.OpenSlideError as error:
            excluded.append(record_exclusion(row,error));continue
        if not images:
            excluded.append(record_exclusion(row,'No tissue-positive patches'));continue
        phi,res=encode(images,models)
        valid=np.unique(groups)
        pooled_phi=np.asarray([phi[groups==g].mean(0) for g in valid],dtype='float32')
        pooled_res=np.asarray([res[groups==g].mean(0) for g in valid],dtype='float32')
        result=bridge.predict_features(pooled_phi,pooled_res)
        auxiliary_rna=auxiliary.predict(pooled_res)
        arrays={'coords_level0':coords[valid],'patch_tissue_fractions':fractions[valid],
            'Phikon':pooled_phi,'ResNet50':pooled_res,'RNA':result['gene_expression'],
            'RNA_C':result['primary_refined_gene_expression'],'programs':result['program_scores'],'RNA_primary_ResNet':auxiliary_rna}
        assert all(np.isfinite(v).all() for v in arrays.values())
        temporary=destination.with_suffix('.partial.npz'); np.savez_compressed(temporary,**arrays); temporary.replace(destination)
        qc.update({'slide_id':row.slide_id,'Patient_ID':row.Patient_ID,'config_sha256':config_hash,
            'valid_FOVs':len(valid),'empty_FOVs':len(coords)-len(valid),'patches_encoded':len(images),
            'median_kept_patches_per_FOV':float(np.median((fractions[valid]>=.05).sum(1))),
            'sampled_tissue_area_um2':float(fractions[valid].mean(1).sum()*np.prod(config['FOV_um'])),
            'source_path':row.path,'source_bytes':stat.st_size,'source_mtime_ns':str(stat.st_mtime_ns),
            'seconds':time.monotonic()-began,'RNA_min':float(arrays['RNA'].min()),'RNA_max':float(arrays['RNA'].max()),
            'device':'cuda','precision':'FP32','RNA_is_image_inferred_not_measured':True})
        write_json(meta,qc)
        successful.append(row.slide_id)
        print(f'{number}/651 {row.slide_id}: {len(valid)} FOVs, {len(images)} patches, {qc["seconds"]:.1f}s; total {(time.monotonic()-started)/60:.1f}min',flush=True)
    pd.DataFrame(excluded,columns=['slide_id','Patient_ID','source_path','reason','status','authorization','original_modified']).to_csv(OUT/'inventory/excluded_WSI_read_errors.csv',index=False)
    write_json(OUT/'features/extraction_complete.json',{'attempted_slides':len(manifest),'successful_slides':len(successful),
        'excluded_slides':len(excluded),'successful_slide_ids':successful,'complete':True,'config_sha256':config_hash})
    print(f'EXTRACTION COMPLETE: {len(successful)} usable / {len(manifest)} attempted; {len(excluded)} excluded with user authorization',flush=True)

if __name__=='__main__': main()
