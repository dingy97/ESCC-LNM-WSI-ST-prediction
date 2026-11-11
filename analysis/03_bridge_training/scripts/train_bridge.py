"""Patient-nested FOV RNA regression. No clinical outcome is loaded or scored."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
from pathlib import Path
from datetime import datetime
from functools import lru_cache
import argparse
import copy
import hashlib
import itertools
import json
import random
import shutil
import time
import traceback
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from bridge_core import patient_mse,patient_weights
from bridge_models import TargetSpace,FeatureSpace,ProgramSpace,BridgeEstimator

R=Path(__file__).resolve().parents[2];T=R/'03_bridge_training'
CONFIG=json.loads((T/'config/stage2_frozen_config.json').read_text(encoding='utf-8'))
SEED=CONFIG['seed'];random.seed(SEED);np.random.seed(SEED)
DATA=None;XDATA={};RUN=None;YTAG='all_cells';QUANTILE=.95;SUPPORT=.1

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def stamp():return datetime.now().astimezone().isoformat()

def load_data(yversion='all_cells'):
    split=pd.read_csv(T/'bridge_FOV_splits.csv').sort_values('FOV_ID').reset_index(drop=True)
    raw=pd.read_csv(T/'input_cache/fov_expression_raw_from_parquet.csv.gz').set_index('FOV_ID').reindex(split.FOV_ID)
    norm=pd.read_csv(T/'input_cache/fov_expression_normalized_from_parquet.csv.gz').set_index('FOV_ID').reindex(split.FOV_ID)
    genes=norm.columns.drop('FOV_key').tolist()
    counts=raw[genes].to_numpy(dtype='int64');y=norm[genes].to_numpy(dtype='float64')
    assert len(genes)==6175 and np.isfinite(y).all()
    assert np.allclose(y,np.log1p(counts/counts.sum(1)[:,None]*1e6),rtol=1e-7,atol=1e-7)
    negative=raw[[g for g in raw.columns if g.startswith('Negative')]].to_numpy(dtype='int64')
    if yversion!='all_cells':
        a=np.load(T/'input_cache/annotation_stratified_RNA.npz')
        assert np.array_equal(a['genes'],genes)
        counts=a['annotated_counts' if yversion=='annotated' else 'unannotated_counts'][split.FOV_ID.to_numpy()-1]
        assert (counts.sum(1)>0).all()
        y=np.log1p(counts/counts.sum(1)[:,None]*1e6)
        controls=np.load(T/'input_cache/annotation_negative_probes.npz')
        negative=controls['annotated' if yversion=='annotated' else 'unannotated'][split.FOV_ID.to_numpy()-1]
    return {'m':split,'y':y,'raw':counts,'negative':negative,'genes':np.array(genes),
            'patients':split.Patient_ID.to_numpy(),'primary':split.is_primary_ESCC.to_numpy(bool),
            'fold':split.outer_fold.to_numpy(int),'FOV':split.FOV_ID.to_numpy(int)}

def feature_data(encoder,aggregation):
    key=(encoder,aggregation)
    if key not in XDATA:
        a=np.load(T/f'features/{encoder}_{aggregation}_FOV_features.npz')
        mapping=pd.Index(a['FOV_ID']).get_indexer(DATA['FOV']);assert (mapping>=0).all()
        x=a['features'][mapping].astype('float64');assert np.isfinite(x).all()
        XDATA[key]=x
    return XDATA[key]

@lru_cache(maxsize=96)
def target_space(indices):
    idx=np.asarray(indices)
    t=TargetSpace(DATA['y'][idx],DATA['raw'][idx],DATA['negative'][idx],DATA['patients'][idx],QUANTILE,SUPPORT)
    ident=hashlib.sha256(np.asarray(indices,dtype='int32').tobytes()).hexdigest()[:16]
    folder=RUN/'training_QC';folder.mkdir(exist_ok=True)
    file=folder/f'{ident}.csv.gz'
    if not file.exists():
        pd.DataFrame({'gene':DATA['genes'],'train_logCPM_mean':t.mean,'train_logCPM_SD':t.scale,
          'training_patient_background_support':t.support,'included':t.mask,'reason':t.reason}).to_csv(file,index=False,compression='gzip')
        dump(folder/f'{ident}.json',{'training_FOVs':DATA['FOV'][idx].tolist(),
           'training_patients':sorted(np.unique(DATA['patients'][idx]).tolist()),
           'genes_pass':int(t.mask.sum()),'RNA_version':YTAG,'negative_quantile':QUANTILE,'patient_support':SUPPORT})
    return t

@lru_cache(maxsize=48)
def feature_space(indices,encoder,aggregation):
    idx=np.asarray(indices)
    return FeatureSpace(feature_data(encoder,aggregation)[idx],DATA['patients'][idx])

@lru_cache(maxsize=144)
def program_space(indices,kind,rank):
    idx=np.asarray(indices)
    return ProgramSpace(kind,rank,target_space(indices),DATA['y'][idx],DATA['patients'][idx])

@lru_cache(maxsize=48)
def base_fit(indices,encoder,aggregation,family,config_json):
    idx=np.asarray(indices);cfg=json.loads(config_json)
    p=program_space(indices,family,cfg['rank']) if family in ['PCA','NMF','Coexpression'] else None
    model=BridgeEstimator(family,cfg,target_space(indices),feature_space(indices,encoder,aggregation),p)
    return model.fit(feature_data(encoder,aggregation)[idx],DATA['y'][idx],DATA['patients'][idx])

def fit_model(train,group,encoder,family,cfg,aggregation='HSV_mean'):
    actual=train[DATA['primary'][train]] if group=='A' else train
    key=tuple(actual.tolist())
    model=base_fit(key,encoder,aggregation,family,json.dumps(cfg,sort_keys=True))
    if group=='C':
        model=copy.deepcopy(model);p=train[DATA['primary'][train]]
        model.specialize(feature_data(encoder,aggregation)[p],DATA['y'][p],DATA['patients'][p])
    return model

def loading_stability(bases):
    values=[]
    for a,b in itertools.combinations(bases,2):
        a=a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
        b=b/np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-12)
        similarities=np.abs(a@b.T);row,col=linear_sum_assignment(-similarities)
        values.append(float(similarities[row,col].mean()))
    return float(np.mean(values)) if values else float('nan')

def inner_splits(train,fold):
    patients=np.unique(DATA['patients'][train]);out=[]
    for a,b in KFold(3,shuffle=True,random_state=SEED+fold).split(patients):
        tr=train[np.isin(DATA['patients'][train],patients[a])]
        va=train[np.isin(DATA['patients'][train],patients[b]) & DATA['primary'][train]]
        assert len(va) and not set(DATA['patients'][tr])&set(DATA['patients'][va])
        out.append((tr,va))
    return out

def run_one(fold,encoder,group,family,aggregation='HSV_mean'):
    name=f'{encoder}_{aggregation}_{group}_{family}'
    folder=RUN/f'outer_fold_{fold}'/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():
        result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
        if result['status']=='COMPLETED':return result
    started=stamp();clock=time.monotonic()
    train=np.flatnonzero(DATA['fold']!=fold);test=np.flatnonzero(DATA['fold']==fold)
    test_primary=test[DATA['primary'][test]]
    heldout=set(DATA['patients'][test]);assert not heldout&set(DATA['patients'][train])
    entries=[];candidates=[]
    for ci,cfg in enumerate(CONFIG['configs'][family]):
        scores=[];recon=[];bases=[];converged=True
        for inner,(tr,va) in enumerate(inner_splits(train,fold)):
            model=fit_model(tr,group,encoder,family,cfg,aggregation)
            assert not set(model.fit_patients)&(set(DATA['patients'][va])|heldout)
            prediction=model.predict(feature_data(encoder,aggregation)[va])
            assert prediction.shape==DATA['y'][va].shape and np.isfinite(prediction).all()
            score=patient_mse(DATA['y'][va],prediction,DATA['patients'][va]);scores.append(score)
            if model.program:
                oracle=model.program.decode(model.program.transform(DATA['y'][va]))
                recon.append(patient_mse(DATA['y'][va],oracle,DATA['patients'][va]));bases.append(model.program.full_loadings())
            converged=converged and bool(model.converged)
            entries.append({'candidate':ci,'inner_fold':inner,'configuration':json.dumps(cfg),
                'primary_patient_MSE':score,'target_reconstruction_MSE':recon[-1] if recon else None,
                'numerical_solver_converged':bool(model.converged),
                'train_patients':json.dumps(model.fit_patients),'validation_patients':json.dumps(sorted(np.unique(DATA['patients'][va]).tolist())),
                'outer_heldout_patients':json.dumps(sorted(heldout))})
        candidates.append({'candidate':ci,'config':cfg,'inner_MSE':float(np.mean(scores)),
            'inner_MSE_SE':float(np.std(scores,ddof=1)/np.sqrt(3)),
            'reconstruction_MSE':float(np.mean(recon)) if recon else None,
            'reconstruction_SE':float(np.std(recon,ddof=1)/np.sqrt(3)) if recon else None,
            'loading_stability':loading_stability(bases) if bases else None,'converged':converged})
    if family in ['PCA','NMF','Coexpression']:
        best=min(candidates,key=lambda c:c['reconstruction_MSE'])
        near=[c for c in candidates if c['reconstruction_MSE']<=best['reconstruction_MSE']+best['reconstruction_SE']]
        stable=[c for c in near if c['loading_stability']>=.7 and c['converged']]
        chosen=min(stable or near,key=lambda c:c['config']['rank'])
    else:chosen=min(candidates,key=lambda c:c['inner_MSE'])
    model=fit_model(train,group,encoder,family,chosen['config'],aggregation)
    prediction=model.predict(feature_data(encoder,aggregation)[test])
    assert np.isfinite(prediction).all()
    primary_train=train[DATA['primary'][train]]
    baseline=np.average(DATA['y'][primary_train],axis=0,weights=patient_weights(DATA['patients'][primary_train]))
    pred_primary=prediction[DATA['primary'][test]]
    mse=patient_mse(DATA['y'][test_primary],pred_primary,DATA['patients'][test_primary])
    bmse=patient_mse(DATA['y'][test_primary],np.tile(baseline,(len(test_primary),1)),DATA['patients'][test_primary])
    np.savez_compressed(folder/'OOF_prediction.npz',FOV_ID=DATA['FOV'][test],prediction=prediction,
                        primary_training_mean=baseline.astype('float32'),gene_mask=model.target.mask)
    joblib.dump(model,folder/'model.joblib',compress=3)
    if model.program:
        np.savez_compressed(folder/'program_loadings_and_scores.npz',genes=DATA['genes'],
            full_loadings=model.program.full_loadings(),training_FOV=DATA['FOV'][train],test_FOV=DATA['FOV'][test],
            test_true_scores=model.program.true_scores(DATA['y'][test]),
            test_predicted_scores=model.latent_predict(model.feature.transform(feature_data(encoder,aggregation)[test]))*model.program.scale+model.program.mean)
    pd.DataFrame(entries).to_csv(folder/'inner_CV.csv',index=False)
    result={'status':'COMPLETED','matrix_id':name,'encoder':encoder,'aggregation':aggregation,
       'group':group,'family':family,'outer_fold':fold,'started_at':started,'ended_at':stamp(),
       'seconds':time.monotonic()-clock,'selected':chosen,'candidates':candidates,
       'primary_MSE':mse,'primary_training_mean_MSE':bmse,'primary_MSE_ratio':mse/bmse,
       'primary_test_patients':len(np.unique(DATA['patients'][test_primary])),
       'primary_test_FOVs':len(test_primary),'all_test_FOVs':len(test),
       'training_patients':model.fit_patients,'heldout_patients':sorted(heldout),
       'QC_genes_passed':int(model.target.mask.sum()),'solver_converged':bool(model.converged),
       'prediction_sha256':sha(folder/'OOF_prediction.npz'),'model_sha256':sha(folder/'model.joblib'),
       'RNA_version':YTAG,'protocol_sha256':sha(T/'config/stage2_frozen_config.json')}
    dump(folder/'result.json',result)
    print(f'fold{fold} {name}: primary MSE ratio={mse/bmse:.4f}, genes={model.target.mask.sum()}, {result["seconds"]:.1f}s',flush=True)
    return result

def main():
    global DATA,RUN,YTAG,QUANTILE,SUPPORT
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['baseline','full','mask_sensitivity'],default='full')
    parser.add_argument('--RNA',choices=['all_cells','annotated','unannotated'],default='all_cells')
    parser.add_argument('--quantile',type=float,default=.95);parser.add_argument('--support',type=float,default=.1)
    parser.add_argument('--only',default='');args=parser.parse_args()
    YTAG=args.RNA;QUANTILE=args.quantile;SUPPORT=args.support
    DATA=load_data(YTAG)
    identity={'protocol':sha(T/'config/stage2_frozen_config.json'),
       'scripts':{n:sha(Path(__file__).with_name(n)) for n in ['bridge_core.py','bridge_models.py','train_bridge.py']},
       'RNA':YTAG,'negative_quantile':QUANTILE,'patient_support':SUPPORT}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
    RUN=T/f'runs/{key}';RUN.mkdir(parents=True,exist_ok=True)
    dump(RUN/'identity.json',identity)
    snapshot=RUN/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for name in identity['scripts']:
        if not (snapshot/name).exists():shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    dump(T/f'active_run_{YTAG}_{args.quantile}_{args.support}.json',{'run':str(RUN),'identity':identity})
    if args.phase=='baseline':models=[('ResNet50','HSV_mean',g,'Ridge') for g in ['A','B','C']]
    elif args.phase=='mask_sensitivity':models=[(e,a,g,'Ridge') for e in CONFIG['encoders'] for a in CONFIG['aggregation_sensitivities'] for g in ['A','B','C']]
    else:models=[(e,'HSV_mean',g,f) for e in CONFIG['encoders'] for g in ['A','B','C'] for f in CONFIG['configs']]
    if args.only:
        e,g,f=args.only.split(':');models=[(e,'HSV_mean',g,f)]
    results=[]
    with threadpool_limits(limits=2):
        for fold in range(5):
            for encoder,aggregation,group,family in models:
                try:results.append(run_one(fold,encoder,group,family,aggregation))
                except Exception as error:
                    name=f'{encoder}_{aggregation}_{group}_{family}'
                    folder=RUN/f'outer_fold_{fold}'/name;folder.mkdir(parents=True,exist_ok=True)
                    dump(folder/'failure.json',{'at':stamp(),'error':repr(error),'traceback':traceback.format_exc(),'no_substitute_predictions_created':True})
                    print(f'FAILED fold{fold} {name}: {error}',flush=True)
                    raise
    simple=[{k:v for k,v in x.items() if not isinstance(v,(dict,list))} for x in results]
    pd.DataFrame(simple).to_csv(RUN/f'fold_results_{args.phase}.csv',index=False)
    dump(RUN/f'completion_{args.phase}.json',{'status':'COMPLETED','ended_at':stamp(),'models':len(models),'outer_runs':len(results)})
    print(f'Completed {args.phase}: {len(results)} outer model fits',flush=True)

if __name__=='__main__':main()
