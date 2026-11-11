"""CUDA Phikon -> frozen Ridge RNA for audited ESD H&E candidates; no model fitting."""
import os,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,time,traceback
import model_space as space
from model_space import ROOT,OUT
import numpy as np,pandas as pd,joblib,openslide,torch
sys.path.insert(0,str(ROOT/'10_lnm_sampling_corrected/scripts'))
import extract_wsi_features as original
from broad_inference import BroadBridgeBundle
from train_all_12 import dump,sha,now

def encode(images,model,batch=24):
 result=[];used=batch
 for start in range(0,len(images),batch):
  current=images[start:start+batch]
  try:
   x=torch.from_numpy(np.stack(current)).permute(0,3,1,2).to('cuda',dtype=torch.float32)/255
   mu=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];sd=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
   with torch.inference_mode():z=model(pixel_values=(x-mu)/sd).last_hidden_state[:,0].cpu().numpy()
   result.append(z);del x
  except torch.cuda.OutOfMemoryError:
   torch.cuda.empty_cache()
   if batch==1:raise
   z,smaller=encode(current,model,max(1,batch//2));result.append(z);used=min(used,smaller)
 return np.concatenate(result),used

def main():
 lock=json.loads((OUT/'config/final_model_lock.json').read_text(encoding='utf-8'))
 assert sha(Path(lock['winner_directory'])/'model.joblib')==lock['model_SHA256']
 assert torch.cuda.is_available(),'GPU unavailable; do not silently change image encoder precision/backend'
 sys.path.insert(0,str(original.OLD/'runtime_deps'))
 from transformers import ViTModel
 torch.set_num_threads(1);torch.manual_seed(20261104);torch.cuda.manual_seed_all(20261104)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
 model=ViTModel.from_pretrained(original.OLD/'weights/phikon',add_pooling_layer=False,local_files_only=True).eval().to('cuda')
 for p in model.parameters():p.requires_grad_(False)
 bundle=BroadBridgeBundle(ROOT/'03_bridge_training/bridge_model_checkpoints/broad_transcriptome')
 config=json.loads((ROOT/'10_lnm_sampling_corrected/config/frozen_analysis.json').read_text(encoding='utf-8'))
 snapshot=OUT/'04_ESD_validation/input_WSI_manifest_snapshot.csv'
 if not snapshot.exists():
  import shutil
  shutil.copy2(OUT/'00_inventory/ESD_WSI_manifest.csv',snapshot)
 frame=pd.read_csv(snapshot,dtype={'case_id':str,'mtime_ns':str})
 frame=frame[frame.eligible_for_current_stage.eq(True)].sort_values(['Patient_ID','filename']).reset_index(drop=True)
 assert len(frame)>0
 contract={'script':sha(__file__),'frozen_sampling':sha(ROOT/'10_lnm_sampling_corrected/config/frozen_analysis.json'),
  'sampler':sha(ROOT/'10_lnm_sampling_corrected/scripts/spatial_sampling.py'),'bridge':sha(ROOT/'03_bridge_training/bridge_model_checkpoints/broad_transcriptome/Phikon_FULL_B_Ridge.joblib'),
  'precision':'FP32','encoder':'frozen local Phikon','batch':24,'FOV_um':config['FOV_um'],'gene_order':hashlib.sha256('|'.join(bundle.genes).encode()).hexdigest()}
 dump(OUT/'config/ESD_prediction_cohort_snapshot.json',{'created_or_checked_at':now(),'manifest':str(snapshot),'SHA256':sha(snapshot),
  'eligible_WSI':len(frame),'purpose':'fixed external identity/label/QC snapshot before risk predictions; image cache contract remains outcome-independent'})
 old_contract=OUT/'config/ESD_extraction_contract.json'
 if old_contract.exists():assert json.loads(old_contract.read_text(encoding='utf-8'))==contract
 else:dump(old_contract,contract)
 directory=OUT/'04_ESD_validation/features';directory.mkdir(parents=True,exist_ok=True)
 qc=[];values=[];durations=[]
 for number,row in enumerate(frame.itertuples(),1):
  key=hashlib.sha256(row.path.encode()).hexdigest()[:24];npz=directory/f'{key}.npz';meta=npz.with_suffix('.json');began=time.monotonic()
  try:
   path=Path(row.path);stat=path.stat();assert stat.st_size==row.bytes and str(stat.st_mtime_ns)==str(row.mtime_ns),'Source file changed after audit'
   if npz.exists() and meta.exists():
    saved=json.loads(meta.read_text(encoding='utf-8'));assert saved['contract']==contract and saved['source_bytes']==row.bytes
    with np.load(npz) as z:value=z['RNA'].mean(0)
    info={**saved,'cache_reused':True}
   elif meta.exists() and json.loads(meta.read_text(encoding='utf-8')).get('status')=='EXCLUDED':
    info=json.loads(meta.read_text(encoding='utf-8'));qc.append(info);continue
   else:
    with openslide.OpenSlide(row.path) as slide:
     mpp=[float(slide.properties[openslide.PROPERTY_NAME_MPP_X]),float(slide.properties[openslide.PROPERTY_NAME_MPP_Y])]
     thumb=slide.get_thumbnail((2048,2048)).convert('RGB')
     mask,maskqc=original.slide_candidate_mask(np.asarray(thumb),slide.dimensions,mpp)
     coords,sampling=original.candidate_fovs(slide.dimensions,mpp,mask,config['FOV_um'],config['max_virtual_FOVs_per_slide'])
     if len(coords)==0:raise ValueError('No tissue FOV candidates under frozen sampling protocol')
     images,groups,fractions=original.read_patches(slide,coords,mpp,config['FOV_um'])
    if not images:raise ValueError('No valid tissue-positive patches')
    phi,used=encode(images,model,24);valid=np.unique(groups)
    pooled=np.asarray([phi[groups==g].mean(0) for g in valid],dtype='float32')
    RNA=bundle.gene_model.predict(np.asarray(pooled,dtype=float));assert RNA.shape[1]==6175 and np.isfinite(RNA).all()
    temporary=npz.with_name(npz.stem+'.partial.npz');np.savez_compressed(temporary,Phikon=pooled,RNA=RNA,coords_level0=coords[valid],patch_tissue_fractions=fractions[valid]);os.replace(temporary,npz)
    value=RNA.mean(0)
    info={'contract':contract,'path':row.path,'filename':row.filename,'Patient_ID':row.Patient_ID,'case_id':row.case_id,'LNM':int(row.LNM_folder),
      'source_bytes':row.bytes,'source_mtime_ns':str(row.mtime_ns),'valid_FOVs':len(valid),'patches':len(images),
      'sampling':sampling,'mask_QC':maskqc,'status':'PASS','device':'cuda','GPU':torch.cuda.get_device_name(0),'minimum_batch_used':used,
      'seconds':time.monotonic()-began,'peak_GPU_MiB':torch.cuda.max_memory_allocated()/2**20,'RNA_is_image_inferred_not_measured':True}
    dump(meta,info);durations.append(info['seconds']);del images,phi,RNA
   values.append({'Patient_ID':row.Patient_ID,'case_id':row.case_id,'filename':row.filename,'LNM':int(row.LNM_folder),'RNA':value})
   qc.append(info)
  except (openslide.OpenSlideError,ValueError,KeyError,AssertionError) as error:
   info={'path':row.path,'filename':row.filename,'Patient_ID':row.Patient_ID,'case_id':row.case_id,'status':'EXCLUDED','reason':str(error),'contract':contract,'original_modified':False};dump(meta,info);qc.append(info)
  dump(OUT/'logs/ESD_extraction_progress.json',{'time':now(),'status':'ENCODING_CUDA','completed':number,'total':len(frame),'usable':len(values),'excluded':sum(z['status']=='EXCLUDED' for z in qc),'estimated_remaining_hours':float(np.median(durations[-20:])*(len(frame)-number)/3600) if durations else None})
  print(now(),'ESD_CUDA',number,'/',len(frame),row.filename,qc[-1]['status'],flush=True)
 pd.DataFrame([{k:v for k,v in z.items() if k not in ['contract','sampling','mask_QC']} for z in qc]).to_csv(OUT/'04_ESD_validation/WSI_extraction_QC.csv',index=False,encoding='utf-8-sig')
 grouped={}
 for z in values:grouped.setdefault(z['Patient_ID'],[]).append(z)
 patient=[];manifest=[]
 overlap_file=OUT/'00_inventory/ESD_CosMx_bridge_overlap_check.json'
 bridge_overlap=set(json.loads(overlap_file.read_text(encoding='utf-8'))['exact_accession_overlap']) if overlap_file.exists() else set()
 for pid,slides in sorted(grouped.items()):
  labels={z['LNM'] for z in slides};assert len(labels)==1
  patient.append(np.asarray([z['RNA'] for z in slides]).mean(0));manifest.append({'Patient_ID':pid,'case_id':slides[0]['case_id'],'LNM':slides[0]['LNM'],'usable_WSI_count':len(slides),'CosMx_bridge_supervision_overlap':slides[0]['case_id'] in bridge_overlap,'validation_unit':'accession_case_group; cross-accession unique-person status not independently established'})
 pd.DataFrame(patient,index=pd.Index([z['Patient_ID'] for z in manifest],name='Patient_ID'),columns=bundle.genes).to_parquet(OUT/'04_ESD_validation/ESD_case_inferred_RNA.parquet')
 pd.DataFrame(manifest).to_csv(OUT/'04_ESD_validation/ESD_analysis_manifest.csv',index=False,encoding='utf-8-sig')
 dump(OUT/'04_ESD_validation/extraction_complete.json',{'time':now(),'attempted_WSI':len(frame),'usable_WSI':len(values),'case_groups':len(manifest),'GPU':torch.cuda.get_device_name(0),'contract':contract,'cohort_status':'restricted H&E candidate case-group evaluation; cannot label full-cohort patient independence verified'})
 print('ESD_EXTRACTION_COMPLETE',len(values),len(manifest),flush=True)

if __name__=='__main__':main()
