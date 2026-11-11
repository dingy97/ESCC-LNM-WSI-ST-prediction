"""Predeclared-issue sensitivities of the fixed candidate; not a new 12-model search."""
import json,hashlib,os
from pathlib import Path
import train_all_12 as t
from model_space import OUT
import numpy as np,pandas as pd,joblib

def run():
 done=OUT/'03_internal_evaluation/sensitivity_complete.json'
 if done.exists():return
 lock=json.loads((OUT/'config/final_model_lock.json').read_text(encoding='utf-8'));name=lock['winner_classifier'];recipe=lock['winner']['recipe']
 t.initialize();data=t.DATA
 def auxiliary_event(classifier,tr,rec,seed,repair=False):
  with (OUT/f'logs/sensitivity_fit_events_{os.getpid()}.jsonl').open('a',encoding='utf-8') as stream:
   stream.write(json.dumps({'time':t.now(),'stage':'fixed-candidate sensitivity only','classifier':classifier,'training_n':len(tr),
    'training_id_digest':hashlib.sha256(data.ids[tr].tobytes()).hexdigest(),'seed':seed,'repair':repair})+'\n')
 t.fit_event=auxiliary_event
 rows=[];cohorts=[('exclude_CosMx_overlap',~data.patients.CosMx_overlap.to_numpy(bool),2000),
                  ('exclude_two_corrected_historical_discrepancies',~data.patients.historical_label_discrepancy.astype(bool).to_numpy(),1000)]
 for cohort,keep,salt in cohorts:
  for fold in range(5):
   directory=OUT/f'03_internal_evaluation/sensitivity/{cohort}/outer_{fold}';directory.mkdir(parents=True,exist_ok=True)
   cached=directory/'predictions.parquet'
   if cached.exists():rows.append(pd.read_parquet(cached));continue
   tr=np.flatnonzero(keep&(data.patients.outer_fold.to_numpy()!=fold));te=np.flatnonzero(keep&(data.patients.outer_fold.to_numpy()==fold))
   inner=t.r.split(data.y[tr],3,20261106+fold+salt)
   model,cp,margin=t.r.calibrated_fit(data,tr,recipe,name,t.r.seedof((20261107,cohort,fold,'fixed_sensitivity')),inner)
   cuts={'0.5':.5,**{str(target):t.r.choose_cutoff(data.y[tr],cp,target) for target in [.95,.98]}}
   p=model.predict_proba(data.x[te])[:,1];records=[]
   for target,cut in cuts.items():records.extend({'cohort':cohort,'Patient_ID':str(data.ids[i]),'LNM':int(data.y[i]),'outer_fold':fold,
      'classifier':name,'target':float(target),'probability':float(v),'threshold':cut,'high_risk':bool(v>=cut)} for i,v in zip(te,p))
   df=pd.DataFrame(records);df.to_parquet(cached,index=False);rows.append(df);joblib.dump(model,directory/'fixed_recipe_model.joblib',compress=3)
   t.dump(directory/'fit_audit.json',{'fit_ids':data.ids[tr].tolist(),'test_ids':data.ids[te].tolist(),'recipe':recipe,
    'selected_genes':data.genes[model.features.genes].tolist(),'calibration_folds':[{'fit_ids':data.ids[tr[a]].tolist(),'holdout_ids':data.ids[tr[b]].tolist()} for a,b in inner],
    'no_new_classifier_or_hyperparameter_selection':True})
   data.feature_cache.clear();data.rank_cache.clear()
  print(t.now(),'FIXED_SENSITIVITY_COMPLETE',cohort,int(keep.sum()),flush=True)
 pd.concat(rows,ignore_index=True).to_parquet(OUT/'03_internal_evaluation/fixed_candidate_sensitivity_OOF.parquet',index=False)
 t.dump(done,{'time':t.now(),'classifier':name,'recipe':recipe,'cohorts':{c:int(k.sum()) for c,k,_ in cohorts},
  'scope':'conditional fixed-candidate sensitivity; not an unbiased re-execution of the entire 12-classifier nomination process; bridge unchanged'})

if __name__=='__main__':run()
