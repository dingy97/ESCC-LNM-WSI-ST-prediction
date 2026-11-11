"""Resumable 12-classifier nested optimization; no external data enter fitting."""
import os,sys
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
import json,hashlib,gzip,time,warnings,traceback
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import model_space as space
from model_space import ROOT,OUT,r
import numpy as np,pandas as pd,joblib
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from sklearn.exceptions import ConvergenceWarning
from filelock import FileLock
from threadpoolctl import threadpool_limits

def now():return datetime.now().astimezone().isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8');os.replace(tmp,p)
def token(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:24]
def status(**x):dump(OUT/'logs/progress.json',{'time':now(),**x})

def metrics(y,p,h):
 y=np.asarray(y,int);h=np.asarray(h,bool);tp=int(np.sum((y==1)&h));tn=int(np.sum((y==0)&~h));fp=int(np.sum((y==0)&h));fn=int(np.sum((y==1)&~h))
 return {'AUROC':float(roc_auc_score(y,p)),'AUPRC':float(average_precision_score(y,p)),'Brier':float(brier_score_loss(y,p)),
 'Accuracy':(tp+tn)/len(y),'Sensitivity':tp/(tp+fn),'Specificity':tn/(tn+fp),'Precision':tp/(tp+fp) if tp+fp else 0.,
 'F1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,'TP':tp,'TN':tn,'FP':fp,'FN':fn,'patients':len(y)}

class SurgicalData(r.Data):
 def __init__(self):
  p=pd.read_csv(OUT/'00_inventory/surgical_patient_manifest.csv',dtype={'case_id':str})
  previous=pd.read_csv(ROOT/'14_RNA_LNM_classifier_optimization/inventory/patient_manifest.csv',dtype={'pathology_case_id':str})
  p=p.merge(previous[['pathology_case_id','CosMx_overlap']],left_on='case_id',right_on='pathology_case_id',validate='one_to_one').sort_values('Patient_ID').reset_index(drop=True)
  f=pd.read_parquet(OUT/'01_feature_mapping/surgical_patient_inferred_RNA.parquet').loc[p.Patient_ID]
  mapping=pd.read_csv(OUT/'01_feature_mapping/patient_feature_reconstruction_audit.csv').set_index('Patient_ID')
  self.patients=p;self.ids=p.Patient_ID.to_numpy(str);self.legacy_ids=mapping.loc[self.ids].legacy_Patient_ID.to_numpy(str)
  self.x=f.to_numpy(float);self.y=p.LNM.to_numpy(int);self.genes=f.columns.to_numpy(str)
  self.fallback=pd.read_csv(ROOT/'14_RNA_LNM_classifier_optimization/inventory/RNA_feature_audit.csv').frozen_training_mean_fallback.to_numpy(bool)
  self.feature_cache={};self.rank_cache={};assert self.x.shape==(160,6175) and np.isfinite(self.x).all()
  assert self.y.sum()==100 and set(self.patients.outer_fold)==set(range(5))
  assert (p.LNM==p.clinical_LNM_verified).all()
 def ranks(self,tr):
  key=hashlib.sha256(self.ids[tr].tobytes()).hexdigest()[:20]
  if key in self.rank_cache:return self.rank_cache[key]
  directory=OUT/'02_training/training_rankings'/key;directory.mkdir(parents=True,exist_ok=True)
  expected={'Patient_ID':self.ids[tr].tolist(),'label_sha256':hashlib.sha256(self.y[tr].tobytes()).hexdigest(),'RNA_sha256':hashlib.sha256(self.x[tr].tobytes()).hexdigest()}
  with FileLock(str(directory/'fit.lock'),timeout=7200):
   if (directory/'rankings.npz').exists():
    meta=json.loads((directory/'fit_patients.json').read_text(encoding='utf-8'));assert all(meta[k]==v for k,v in expected.items())
    with np.load(directory/'rankings.npz') as z:ranks=dict(z)
   else:
    oldkey=hashlib.sha256(self.legacy_ids[tr].tobytes()).hexdigest()[:20];source=ROOT/'14_RNA_LNM_classifier_optimization/models/training_rankings'/oldkey
    meta=json.loads((source/'fit_patients.json').read_text(encoding='utf-8')) if (source/'fit_patients.json').exists() else {}
    valid_old=meta.get('Patient_ID')==self.legacy_ids[tr].tolist() and meta.get('label_sha256')==expected['label_sha256'] and meta.get('RNA_sha256')==expected['RNA_sha256']
    if valid_old:
     with np.load(source/'rankings.npz') as z:ranks=dict(z)
     provenance={'source':'exact training-node Stage14 cache, labels/RNA/identity rechecked','source_path':str(source),'source_SHA256':sha(source/'rankings.npz')}
    else:
     valid=(~self.fallback)&(~r.near_constant_columns(self.x[tr]));ranks,extra=r.nine_rankings(self.x[tr],self.y[tr],valid,r.seedof(key));provenance={'source':'Stage21 train-only ranking refit',**extra}
    temp=directory/f'rankings.{os.getpid()}.npz';np.savez_compressed(temp,**ranks);os.replace(temp,directory/'rankings.npz')
    dump(directory/'fit_patients.json',{**expected,**provenance})
  self.rank_cache[key]=ranks;return ranks

DATA=None;PLAN=None;FITS=0;REPAIRS=0;EXEC=None;JOURNAL=None
def fit_event(name,tr,recipe,seed,repair=False):
 global JOURNAL
 if JOURNAL is None:JOURNAL=(OUT/f'logs/classifier_fit_events_{os.getpid()}.jsonl').open('a',encoding='utf-8',buffering=1)
 JOURNAL.write(json.dumps({'event':'completed_classifier_fit','time':now(),'classifier':name,'training_n':len(tr),
   'training_id_digest':hashlib.sha256(DATA.ids[tr].tobytes()).hexdigest(),'recipe_id':token(recipe),'seed':seed,'repair':repair})+'\n')
def fit_raw(data,tr,recipe,name,seed):
 global FITS,REPAIRS
 f,z=data.features(tr,recipe);m=space.build(name,recipe['config'],seed)
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always');warnings.filterwarnings('ignore',category=FutureWarning);warnings.filterwarnings('ignore',message='X does not have valid feature names')
  m.fit(z[tr],data.y[tr]);FITS+=1;fit_event(name,tr,recipe,seed)
 if any(issubclass(w.category,ConvergenceWarning) for w in caught) and name in ['MLP','Logistic Regression']:
  original=[str(w.message) for w in caught];m=space.build(name,recipe['config'],seed)
  m.set_params(max_iter=24000 if name=='MLP' else 40000)
  if name=='MLP':m.set_params(max_fun=240000)
  with warnings.catch_warnings(record=True) as retry:
   warnings.simplefilter('always');warnings.filterwarnings('ignore',category=FutureWarning);m.fit(z[tr],data.y[tr]);FITS+=1;REPAIRS+=1;fit_event(name,tr,recipe,seed,True)
  with (OUT/f'logs/numerical_repairs_{os.getpid()}.jsonl').open('a',encoding='utf-8') as stream:
   stream.write(json.dumps({'time':now(),'classifier':name,'fit_ids':data.ids[tr].tolist(),'recipe':recipe,'seed':seed,'original_warnings':original,'retry_warnings':[str(w.message) for w in retry],'unresolved':any(issubclass(w.category,ConvergenceWarning) for w in retry)},ensure_ascii=False)+'\n')
  if any(issubclass(w.category,ConvergenceWarning) for w in retry):
   raise RuntimeError('Numerical convergence remains unresolved after uniform repair; preserve trial and inspect')
 return f,m,z

def initialize():
 global DATA,PLAN,EXEC
 PLAN=json.loads((OUT/'config/frozen_12_classifier_plan.json').read_text(encoding='utf-8'))
 EXEC=json.loads((OUT/'config/execution_contract.json').read_text(encoding='utf-8'))
 r.OUT=OUT;r.publish=lambda **kwargs:None;r.fit_raw=fit_raw
 warnings.filterwarnings('ignore',message='X does not have valid feature names')
 DATA=SurgicalData()

def order(row):
 return (-row['inner_mean_AUROC'],-row['inner_mean_AUPRC'],row['Brier'],row['inner_SD_AUROC'],row['recipe']['count'],token({'classifier':row['classifier'],'recipe':row['recipe']}))

def evaluate_trial(c,name,recipe,d):
 global FITS,REPAIRS
 key=token(recipe);path=d/'trials'/f'{key}.json.gz'
 if path.exists():
  with gzip.open(path,'rt',encoding='utf-8') as f:saved=json.load(f)
  assert saved['execution_identity']==EXEC and saved['recipe']==recipe
  return saved['summary']
 before=FITS;repair_before=REPAIRS;data=DATA;tr=np.array(c['tr'],int)
 inner=r.split(data.y[tr],3,20261106+c['fold']+c['cohort_seed']);probs=np.full(len(tr),np.nan)
 high={t:np.zeros(len(tr),bool) for t in [.95,.98]};evidence=[];folds=[]
 for j,(a,b) in enumerate(inner):
  train=tr[a];val=tr[b];sub=r.split(data.y[train],2,20261106+j+100+c['cohort_seed']+c['fold'])
  assert not set(data.ids[train])&set(data.ids[val]);assert not set(data.ids[tr])&set(data.ids[np.array(c['te'],int)])
  model,cp,margin=r.calibrated_fit(data,train,recipe,name,r.seedof((20261107,c['cohort'],c['fold'],j)),sub)
  pp=model.predict_proba(data.x[val])[:,1];assert np.isfinite(pp).all();probs[b]=pp
  cuts={str(t):r.choose_cutoff(data.y[train],cp,t) for t in [.95,.98]}
  for t in [.95,.98]:high[t][b]=pp>=cuts[str(t)]
  mm=metrics(data.y[val],pp,pp>=.5);folds.append(mm)
  evidence.append({'inner_fold':j,'fit_ids':data.ids[train].tolist(),'validation_ids':data.ids[val].tolist(),
    'calibration_CV':[{'fit_ids':data.ids[train[u]].tolist(),'holdout_ids':data.ids[train[v]].tolist()} for u,v in sub],
    'calibration_margins':margin.tolist(),'calibration_probabilities':cp.tolist(),'validation_probabilities':pp.tolist(),
    'thresholds':cuts,'gene_indices':model.features.genes.tolist()})
 assert np.isfinite(probs).all()
 summary={'classifier':name,'recipe':recipe,'candidate_id':key,**metrics(data.y[tr],probs,probs>=.5),
  'inner_mean_AUROC':float(np.mean([z['AUROC'] for z in folds])),'inner_SD_AUROC':float(np.std([z['AUROC'] for z in folds],ddof=1)),
  'inner_mean_AUPRC':float(np.mean([z['AUPRC'] for z in folds])),'clinical_workpoints':{str(t):metrics(data.y[tr],probs,high[t]) for t in [.95,.98]},
  'classifier_base_fit_calls':FITS-before,'numerical_repair_fit_calls':REPAIRS-repair_before}
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
 with gzip.open(tmp,'wt',encoding='utf-8') as stream:json.dump({'execution_identity':EXEC,'recipe':recipe,'summary':summary,'inner_evidence':evidence,'fold_metrics':folds},stream,ensure_ascii=False)
 os.replace(tmp,path)
 return summary

def unit(job):
 with threadpool_limits(limits=1):return _unit(job)

def _unit(job):
 start=time.monotonic();c=job['context'];name=job['classifier'];d=OUT/f'02_training/{c["cohort"]}/outer_{c["fold"]}/{name.replace(" ","_")}'
 d.mkdir(parents=True,exist_ok=True)
 if (d/'complete.json').exists():
  done=json.loads((d/'complete.json').read_text(encoding='utf-8'));assert done['execution_identity']==EXEC;return done
 rows=[];attempts=[]
 def attempt(recipe,stage):
  before=FITS;repair_before=REPAIRS
  try:
   row=evaluate_trial(c,name,recipe,d);rows.append(row);attempts.append({'attempt':len(attempts)+1,'stage':stage,'status':'valid',**row})
  except RuntimeError as error:
   if 'Numerical convergence remains unresolved' not in str(error):raise
   record={'attempt':len(attempts)+1,'stage':stage,'status':'invalid_numerical_QC','classifier':name,'recipe':recipe,'candidate_id':token(recipe),'reason':str(error),
     'classifier_base_fit_calls':FITS-before,'numerical_repair_fit_calls':REPAIRS-repair_before}
   dump(d/'invalid_trials'/f'{token(recipe)}.json',record);attempts.append(record);row=None
  if len(attempts)%6==0:dump(d/'progress.json',{'time':now(),'classifier':name,'context':c['cohort'],'fold':c['fold'],'attempts_done':len(attempts),'attempts_total':192,'stage':stage})
  return row
 for recipe in space.coarse(name):attempt(recipe,'A_complete_factorial')
 best_by_selector=[min([row for row in rows if row['recipe']['selector']==s],key=order) for s in space.SELECTORS if any(row['recipe']['selector']==s for row in rows)]
 assert len(best_by_selector)>=2,'Fewer than two numerically valid selectors; cannot silently omit a classifier'
 top=sorted(best_by_selector,key=order)[:2];draws=PLAN['fine_draws'][f'{c["cohort"]}_{c["fold"]}'][name]
 position=0
 for selector in [z['recipe']['selector'] for z in top]:
  for k in space.COUNTS:
   for rep in range(2):attempt({'selector':selector,'count':k,'config':draws[position]},'B_all_classifiers_refined');position+=1
 best=min(rows,key=order);matched=[]
 for k in space.COUNTS:
  recipe={**best['recipe'],'count':k};result=attempt(recipe,'C_matched_count_comparison');matched.append(result if result is not None else {'recipe':recipe,'status':'invalid_numerical_QC'})
 assert len(attempts)==192
 chosen=min(rows,key=order);data=DATA;tr=np.array(c['tr'],int);te=np.array(c['te'],int)
 inner=r.split(data.y[tr],3,20261106+c['fold']+c['cohort_seed'])
 model,cp,margin=r.calibrated_fit(data,tr,chosen['recipe'],name,r.seedof((20261107,c['cohort'],c['fold'],'final')),inner)
 cuts={str(t):r.choose_cutoff(data.y[tr],cp,t) for t in [.95,.98]}
 joblib.dump(model,d/'model.joblib',compress=3)
 selected={**chosen,'selected_genes':data.genes[model.features.genes].tolist(),'thresholds':{'0.5':.5,**cuts},
  'calibration_parameters':{'method':'training crossfitted sigmoid C=1','coef':model.calibrator.coef_.tolist(),'intercept':model.calibrator.intercept_.tolist()},
  'attempted_searches':192,'valid_search_attempts':len(rows),'invalid_numerical_attempts':sum(x['status']!='valid' for x in attempts),'unique_valid_trials':len({x['candidate_id'] for x in rows}),
  'search_classifier_fit_calls':sum(x['classifier_base_fit_calls'] for x in {z['candidate_id']:z for z in rows}.values())+sum(x['classifier_base_fit_calls'] for x in attempts if x['status']!='valid'),
  'fit_patient_ids':data.ids[tr].tolist(),'test_patient_ids':data.ids[te].tolist(),'model_SHA256':sha(d/'model.joblib')}
 dump(d/'chosen.json',selected);dump(d/'all_attempts.json',attempts);dump(d/'matched_count_curve.json',matched)
 pd.DataFrame({'Patient_ID':data.ids[tr],'LNM':data.y[tr],'raw_margin':margin,'calibration_probability':cp}).to_parquet(d/'training_calibration.parquet',index=False)
 prediction=[]
 if len(te):
  pp=model.predict_proba(data.x[te])[:,1]
  for target,cut in [('0.5',.5),*cuts.items()]:
   prediction.extend({'Patient_ID':str(data.ids[i]),'LNM':int(data.y[i]),'outer_fold':c['fold'],'classifier':name,
     'target':float(target),'probability':float(p),'threshold':cut,'high_risk':bool(p>=cut)} for i,p in zip(te,pp))
 pd.DataFrame(prediction).to_parquet(d/'outer_predictions.parquet',index=False)
 done={'classifier':name,'context':c,'directory':str(d),'seconds':time.monotonic()-start,'execution_identity':EXEC,'chosen':selected,'completed_at':now()}
 dump(d/'complete.json',done);data.feature_cache.clear();data.rank_cache.clear()
 print(now(),'UNIT_COMPLETE',c['cohort'],c['fold'],name,'attempts',192,'unique',selected['unique_valid_trials'],'inner_AUC',round(selected['inner_mean_AUROC'],4),'seconds',round(done['seconds'],1),flush=True)
 return done

def run():
 global PLAN,EXEC
 PLAN=json.loads((OUT/'config/frozen_12_classifier_plan.json').read_text(encoding='utf-8'))
 assert json.loads((OUT/'00_inventory/audit_summary.json').read_text(encoding='utf-8'))['surgical_training_gate']
 for p,h in PLAN['input_hashes'].items():assert sha(p)==h,p
 identity={'plan_sha256':sha(OUT/'config/frozen_12_classifier_plan.json'),
   'training_code':sha(__file__),'model_space':sha(OUT/'scripts/model_space.py'),
   'source_training_functions':sha(ROOT/'14_RNA_LNM_classifier_optimization/scripts/rna_experiment.py')}
 contract=OUT/'config/execution_contract.json'
 if contract.exists():assert json.loads(contract.read_text(encoding='utf-8'))==identity,'Do not overwrite a changed execution contract'
 else:dump(contract,identity)
 initialize();data=DATA;jobs=[]
 for c in PLAN['contexts']:
  tr=np.arange(len(data.y)) if c['cohort']=='full_development' else np.flatnonzero(data.patients.outer_fold.to_numpy()!=c['fold'])
  te=np.array([],int) if c['cohort']=='full_development' else np.flatnonzero(data.patients.outer_fold.to_numpy()==c['fold'])
  context={**c,'tr':tr.tolist(),'te':te.tolist()}
  jobs.extend({'context':context,'classifier':n} for n in space.CLASSIFIERS)
 dump(OUT/'private/training_jobs.json',jobs)
 completed=[];start=time.monotonic()
 status(status='TRAINING_ALL_12',completed_units=0,total_units=len(jobs),search_attempts_per_unit=192)
 with ProcessPoolExecutor(max_workers=PLAN['parallel_workers'],initializer=initialize) as pool:
  pending={pool.submit(unit,j):j for j in jobs}
  while pending:
   done,_=wait(pending,timeout=20,return_when=FIRST_COMPLETED)
   for future in done:completed.append(future.result());pending.pop(future)
   status(status='TRAINING_ALL_12',completed_units=len(completed),total_units=len(jobs),elapsed_hours=(time.monotonic()-start)/3600,search_attempts_per_unit=192)
 dump(OUT/'02_training/all_units_complete.json',completed)
 final=[z for z in completed if z['context']['cohort']=='full_development']
 assert len(final)==12;ranked=sorted(final,key=lambda z:order(z['chosen']));winner=ranked[0]
 lock={'locked_at':now(),'selection_rule':PLAN['selection_rule'],'selection_scope':'full-development INNER scores only, no outer/ESD/TCGA selection',
  'winner_classifier':winner['classifier'],'winner_directory':winner['directory'],'winner':winner['chosen'],
  'ranking':[{'rank':i+1,'classifier':z['classifier'],'directory':z['directory'],'chosen':z['chosen']} for i,z in enumerate(ranked)],
  'historical_exploration_known':True,'frozen_historical_model_replaced':False,'model_SHA256':winner['chosen']['model_SHA256']}
 dump(OUT/'config/final_model_lock.json',lock)
 main=[z for z in completed if z['context']['cohort']=='main'];all_predictions=[];process_predictions=[]
 for f in range(5):
  units=[z for z in main if z['context']['fold']==f];assert len(units)==12;best=min(units,key=lambda z:order(z['chosen']))
  for z in units:all_predictions.append(pd.read_parquet(Path(z['directory'])/'outer_predictions.parquet'))
  one=pd.read_parquet(Path(best['directory'])/'outer_predictions.parquet');one['selection_process_classifier']=best['classifier'];process_predictions.append(one)
 pd.concat(all_predictions,ignore_index=True).to_parquet(OUT/'03_internal_evaluation/all_12_patient_OOF.parquet',index=False)
 pd.concat(process_predictions,ignore_index=True).to_parquet(OUT/'03_internal_evaluation/complete_selection_process_OOF.parquet',index=False)
 for p,h in PLAN['input_hashes'].items():assert sha(p)==h,p
 dump(OUT/'02_training/training_complete.json',{'time':now(),'units':len(completed),'classifiers':12,'attempts_per_unit':192,'total_attempts':sum(z['chosen']['attempted_searches'] for z in completed),
  'unique_trials':sum(z['chosen']['unique_valid_trials'] for z in completed),'classifier_base_search_fit_calls':sum(z['chosen']['search_classifier_fit_calls'] for z in completed),
  'elapsed_hours':(time.monotonic()-start)/3600,'winner':winner['classifier'],'source_inputs_unchanged':True})
 status(status='TRAINING_COMPLETE_ANALYSIS_PENDING',completed_units=72,total_units=72,winner=winner['classifier'])

if __name__=='__main__':
 try:run()
 except Exception:
  dump(OUT/f'logs/training_failure_{int(time.time())}.json',{'time':now(),'traceback':traceback.format_exc(),'status':'FAILED_RESUMABLE'})
  raise
