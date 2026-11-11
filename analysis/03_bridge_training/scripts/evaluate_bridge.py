"""Read complete OOF artifacts, evaluate patient endpoints, freeze deployment fits."""
from pathlib import Path
from datetime import datetime
import json
import itertools
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata,t
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits
from bridge_core import patient_weights,patient_mse,safe_correlations
import train_bridge as tr

R=tr.R;T=tr.T;G=R/'04_gene_predictability';P=R/'05_molecular_programs'

def bh(values):
    result=np.full(len(values),np.nan);ok=np.isfinite(values)
    if ok.any():result[ok]=multipletests(np.asarray(values)[ok],method='fdr_bh')[1]
    return result

def unit_arrays(y,p,b,groups):
    units=np.unique(groups)
    return (units,np.stack([y[groups==u].mean(0) for u in units]),
        np.stack([p[groups==u].mean(0) for u in units]),
        np.stack([((y-p)[groups==u]**2).mean(0) for u in units]),
        np.stack([((y-b)[groups==u]**2).mean(0) for u in units]))

def ci_ratio(error,baseline,repeats=2000):
    rng=np.random.default_rng(20261104)
    ix=rng.integers(0,len(error),size=(repeats,len(error)))
    ratios=error[ix].mean(1)/baseline[ix].mean(1)
    return np.quantile(ratios,[.025,.975]).tolist()

def read_predictions(run,data,aggregation='HSV_mean'):
    results=[]
    for f in sorted(run.glob('outer_fold_*/*/result.json')):
        rec=json.loads(f.read_text(encoding='utf-8'))
        if rec['aggregation']==aggregation:results.append((f.parent,rec))
    expected=42*5 if aggregation=='HSV_mean' else 6*5
    assert len(results)==expected,f'Incomplete model matrix: {len(results)}/{expected}'
    models={};baseline=np.zeros_like(data['y'],dtype='float32')
    for folder,rec in results:
        name=rec['matrix_id']
        if name not in models:models[name]={'pred':np.full_like(data['y'],np.nan,dtype='float32'),'records':[],
           'mask':np.ones(len(data['genes']),bool),'folders':[]}
        a=np.load(folder/'OOF_prediction.npz');pos=pd.Index(data['FOV']).get_indexer(a['FOV_ID'])
        assert (pos>=0).all() and np.isnan(models[name]['pred'][pos]).all()
        models[name]['pred'][pos]=a['prediction'];models[name]['records'].append(rec)
        models[name]['mask'] &= a['gene_mask'];models[name]['folders'].append(folder)
        baseline[pos]=a['primary_training_mean']
    for model in models.values():assert np.isfinite(model['pred']).all() and len(model['records'])==5
    return models,baseline

def model_summary(name,entry,data,baseline):
    selected=data['primary'];groups=data['patients'][selected]
    units,yy,pp,errors,be=unit_arrays(data['y'][selected],entry['pred'][selected],baseline[selected],groups)
    error=errors.mean(1);base=be.mean(1);lo,hi=ci_ratio(error,base)
    row=entry['records'][0]
    return {'model_id':name,'encoder':row['encoder'],'group':row['group'],'family':row['family'],
       'primary_patient_MSE':float(error.mean()),'primary_baseline_MSE':float(base.mean()),
       'primary_MSE_ratio':float(error.mean()/base.mean()),'ratio_CI_low':lo,'ratio_CI_high':hi,
       'primary_RMSE':float(np.sqrt(error.mean())),
       'primary_MSE_patient_SE':float(error.std(ddof=1)/np.sqrt(len(error))),
       'primary_patients':len(units),'primary_FOVs':int(selected.sum()),
       'median_patient_Spearman_all_genes':float(np.nanmedian(safe_correlations(rankdata(yy,axis=0),rankdata(pp,axis=0)))),
       'QC_genes_all_outer_training_folds':int(entry['mask'].sum()),
       'solver_converged_all_folds':all(rec['solver_converged'] for rec in entry['records']),
       'positive_improvement_folds':sum(rec['primary_MSE_ratio']<1 for rec in entry['records'])}

def gene_metrics(y,p,b,patients,folds,genes,mask,scope,model_id,raw):
    units,yy,pp,errors,be=unit_arrays(y,p,b,patients)
    r=safe_correlations(yy,pp);rho=safe_correlations(rankdata(yy,axis=0),rankdata(pp,axis=0))
    denom=np.maximum(1-rho*rho,1e-15)
    pvalue=2*t.sf(np.abs(rho)*np.sqrt((len(units)-2)/denom),df=len(units)-2)
    positive=np.zeros(len(genes),int);valid=np.zeros(len(genes),int)
    for fold in np.unique(folds):
        use=folds==fold;g=patients[use];uy=np.unique(g)
        if len(uy)<3:continue
        a=np.stack([y[use][g==u].mean(0) for u in uy]);bb=np.stack([p[use][g==u].mean(0) for u in uy])
        cor=safe_correlations(rankdata(a,axis=0),rankdata(bb,axis=0))
        positive+=np.isfinite(cor)&(cor>0);valid+=np.isfinite(cor)
    mse=errors.mean(0);bmse=be.mean(0)
    mae=np.stack([np.abs(y[patients==u]-p[patients==u]).mean(0) for u in units]).mean(0)
    variance=np.average((y-yy.mean(0))**2,axis=0,weights=patient_weights(patients))
    ratio=np.divide(mse,bmse,out=np.full(len(genes),np.nan),where=bmse>1e-12)
    q=bh(pvalue)
    stable=(rho>0)&(q<.05)&(ratio<1)&(positive>=4)&mask&(len(units)>=20)
    out=pd.DataFrame({'gene':genes,'model_id':model_id,'scope':scope,
       'Pearson_FOV':safe_correlations(y,p),'Spearman_FOV':safe_correlations(rankdata(y,axis=0),rankdata(p,axis=0)),
       'Pearson_patient_mean':r,'Spearman_patient_mean':rho,'Spearman_patient_p_approx':pvalue,'Spearman_patient_BH_q':q,
       'MAE_patient_balanced':mae,'RMSE_patient_balanced':np.sqrt(mse),'MSE_ratio_training_mean':ratio,
       'R2_patient_balanced':np.divide(variance-mse,variance,out=np.full(len(genes),np.nan),where=variance>1e-12),
       'FOVs':len(y),'patients':len(units),'positive_patient_Spearman_folds':positive,'evaluable_folds':valid,
       'QC_pass_every_training_fold':mask,'stable_exploratory_candidate':stable,
       'mean_logCPM':np.average(y,axis=0,weights=patient_weights(patients)),
       'FOV_detection_fraction':(raw>0).mean(0),
       'correlation_status':np.where(np.isfinite(rho),'evaluable','constant_truth_or_prediction')})
    return out,errors,be

def all_patient_details(y,p,b,patients,genes,scope):
    frames=[]
    for u in np.unique(patients):
        use=patients==u;n=int(use.sum())
        frames.append(pd.DataFrame({'Patient_ID':u,'scope':scope,'gene':genes,'FOVs':n,
           'truth_mean':y[use].mean(0),'OOF_prediction_mean':p[use].mean(0),
           'MAE':np.abs(y[use]-p[use]).mean(0),'RMSE':np.sqrt(((y[use]-p[use])**2).mean(0)),
           'baseline_RMSE':np.sqrt(((y[use]-b[use])**2).mean(0)),
           'within_patient_Spearman':safe_correlations(rankdata(y[use],axis=0),rankdata(p[use],axis=0)) if n>=3 else np.full(len(genes),np.nan),
           'within_patient_correlation_status':'exploratory_spatial_FOV_association' if n>=3 else 'not_computed_fewer_than_3_FOVs'}))
    return frames

def choose_fixed(table):
    eligible=table.loc[table.solver_converged_all_folds].copy()
    best=eligible.loc[eligible.primary_patient_MSE.idxmin()]
    near=eligible.loc[eligible.primary_patient_MSE<=best.primary_patient_MSE+best.primary_MSE_patient_SE].copy()
    near['complexity']=near.family.map({x:i for i,x in enumerate(tr.CONFIG['complexity_order'])})
    near['encoder_order']=near.encoder.map({'ResNet50':0,'Phikon':1})
    near['group_order']=near.group.map({'A':0,'B':1,'C':2})
    return near.sort_values(['complexity','encoder_order','group_order','primary_patient_MSE']).iloc[0]

def main():
    data=tr.load_data();tr.DATA=data
    active=json.loads((T/'active_run_all_cells_0.95_0.1.json').read_text(encoding='utf-8'));run=Path(active['run']);tr.RUN=run
    models,baseline=read_predictions(run,data)
    table=pd.DataFrame([model_summary(n,e,data,baseline) for n,e in models.items()])
    fixed=choose_fixed(table);fixed_id=fixed.model_id
    nested=np.zeros_like(data['y'],dtype='float32');nested_mask=np.ones(len(data['genes']),bool);choices=[]
    for fold in range(5):
        candidates=[(n,next(rec for rec in e['records'] if rec['outer_fold']==fold)) for n,e in models.items()]
        conv=[pair for pair in candidates if pair[1]['selected']['converged'] and pair[1]['solver_converged']]
        best_name,record=min(conv or candidates,key=lambda pair:pair[1]['selected']['inner_MSE'])
        pos=data['fold']==fold;nested[pos]=models[best_name]['pred'][pos]
        folder=next(folder for folder in models[best_name]['folders'] if f'outer_fold_{fold}' in str(folder))
        nested_mask &= np.load(folder/'OOF_prediction.npz')['gene_mask']
        choices.append({'outer_fold':fold,'selected_model':best_name,'encoder':record['encoder'],'group':record['group'],
           'family':record['family'],'inner_primary_MSE':record['selected']['inner_MSE'],
           'outer_primary_MSE_ratio':record['primary_MSE_ratio'],'selected_configuration':json.dumps(record['selected']['config'])})
    pd.DataFrame(choices).to_csv(T/'nested_selection_by_fold.csv',index=False)
    nested_entry={'pred':nested,'mask':nested_mask,'records':[dict(models[choices[f]['selected_model']]['records'][0],
        encoder='nested_selected',group='nested_selected',family='nested_selected',solver_converged=True,
        primary_MSE_ratio=choices[f]['outer_primary_MSE_ratio']) for f in range(5)]}
    ns=model_summary('NESTED_SELECTOR',nested_entry,data,baseline)
    table=pd.concat([table,pd.DataFrame([ns])],ignore_index=True)
    table.to_csv(T/'bridge_model_comparison.csv',index=False)
    encoder=table.loc[table.family.eq('Ridge')].copy();encoder['comparison_scope']='matched_Ridge_same_group_same_folds'
    encoder.to_csv(T/'encoder_comparison.csv',index=False)
    # Whole-panel audit is descriptive. Fold-specific filters remain in each run's training_QC.
    fullqc=pd.read_csv(R/'02_expression/full_panel_feature_qc.csv').query("feature_type=='RNA_gene'")
    fullqc=fullqc.set_index('feature').reindex(data['genes']).reset_index().rename(columns={'feature':'gene'})
    fullqc['eligible_FOV_zero_fraction']=(data['raw']==0).mean(0)
    fullqc['eligible_FOV_mean_logCPM']=data['y'].mean(0);fullqc['eligible_FOV_variance_logCPM']=data['y'].var(0,ddof=1)
    fullqc['scope']='Descriptive full panel; no all-cohort fitted filter used in CV'
    fullqc.to_csv(G/'gene_qc.csv',index=False)
    detailed=[];all_models=[];summarygenes={}
    for name,entry in [('NESTED_SELECTOR',nested_entry),(fixed_id,models[fixed_id])]:
        parts=[]
        for scope,use in [('primary',data['primary']),('broad',np.ones(len(data['y']),bool))]:
            frame,errors,be=gene_metrics(data['y'][use],entry['pred'][use],baseline[use],data['patients'][use],
                data['fold'][use],data['genes'],entry['mask'],scope,name,data['raw'][use])
            if scope=='primary':
                rng=np.random.default_rng(20261104);counts=rng.multinomial(len(errors),np.repeat(1/len(errors),len(errors)),size=2000)
                ratios=(counts@errors)/np.maximum(counts@be,1e-12)
                frame['MSE_ratio_CI_low']=np.quantile(ratios,.025,axis=0);frame['MSE_ratio_CI_high']=np.quantile(ratios,.975,axis=0)
            parts.append(frame)
            if name=='NESTED_SELECTOR':detailed+=all_patient_details(data['y'][use],entry['pred'][use],baseline[use],data['patients'][use],data['genes'],scope)
        summarygenes[name]=pd.concat(parts,ignore_index=True)
        summarygenes[name].to_csv(G/('gene_predictability.csv' if name=='NESTED_SELECTOR' else 'fixed_deployment_gene_predictability.csv'),index=False)
    pd.concat(detailed,ignore_index=True).to_csv(G/'gene_predictability_by_patient.csv',index=False)
    # Retain all candidates' full OOF matrices; none are replaced with only selected genes.
    out=T/'OOF_predictions_per_model';out.mkdir(exist_ok=True)
    for name,entry in models.items():
        np.savez_compressed(out/f'{name}.npz',FOV_ID=data['FOV'],genes=data['genes'],prediction=entry['pred'])
        frame,_,_=gene_metrics(data['y'][data['primary']],entry['pred'][data['primary']],baseline[data['primary']],
             data['patients'][data['primary']],data['fold'][data['primary']],data['genes'],entry['mask'],'primary',name,data['raw'][data['primary']])
        all_models.append(frame)
    pd.concat(all_models,ignore_index=True).to_csv(G/'all_models_gene_predictability.csv.gz',index=False,compression='gzip')
    np.savez_compressed(T/'main_nested_OOF.npz',FOV_ID=data['FOV'],genes=data['genes'],prediction=nested,baseline=baseline,gene_mask=nested_mask)
    np.savez_compressed(T/'fixed_family_OOF.npz',FOV_ID=data['FOV'],genes=data['genes'],prediction=models[fixed_id]['pred'],baseline=baseline,gene_mask=models[fixed_id]['mask'])
    for name,pred in [('bridge_oof_predictions',nested),('fixed_family_oof_predictions',models[fixed_id]['pred'])]:
        frame=pd.DataFrame(pred,columns=data['genes']);
        for col in ['outer_fold','Patient_ID','FOV_key','FOV_ID']:
            source=data['m']['outer_fold' if col=='outer_fold' else col].to_numpy()
            frame.insert(0,col,source)
        frame.to_csv(T/f'{name}.csv.gz',index=False,compression='gzip')
    # Per-patient performance is the unit of strategy comparisons and uncertainty.
    ptrows=[]
    for name,entry in list(models.items())+[('NESTED_SELECTOR',nested_entry)]:
        use=data['primary'];units,_,_,errors,be=unit_arrays(data['y'][use],entry['pred'][use],baseline[use],data['patients'][use])
        for i,u in enumerate(units):ptrows.append({'model_id':name,'Patient_ID':u,'MSE':errors[i].mean(),
                 'baseline_MSE':be[i].mean(),'MSE_ratio':errors[i].mean()/be[i].mean()})
    pt=pd.DataFrame(ptrows);pt.to_csv(T/'primary_per_patient_model_errors.csv',index=False)
    contrasts=[]
    rng=np.random.default_rng(20261104)
    for encoder,family in itertools.product(tr.CONFIG['encoders'],tr.CONFIG['configs']):
        for a,b in [('A','B'),('A','C'),('B','C')]:
            first=pt.loc[pt.model_id.eq(f'{encoder}_HSV_mean_{a}_{family}')].set_index('Patient_ID')
            second=pt.loc[pt.model_id.eq(f'{encoder}_HSV_mean_{b}_{family}')].set_index('Patient_ID').reindex(first.index)
            delta=second.MSE.to_numpy()-first.MSE.to_numpy();base=first.baseline_MSE.to_numpy()
            ix=rng.integers(0,len(delta),size=(5000,len(delta)))
            values=delta[ix].mean(1)/base[ix].mean(1)
            signs=rng.choice([-1.,1.],size=(10000,len(delta)))
            pvalue=(1+(np.abs(signs@delta/len(delta))>=abs(delta.mean())).sum())/(len(signs)+1)
            contrasts.append({'encoder':encoder,'family':family,'contrast':b+' minus '+a,
                'delta_MSE_over_common_baseline':delta.mean()/base.mean(),
                'CI_low':np.quantile(values,.025),'CI_high':np.quantile(values,.975),
                'paired_patient_signflip_p':pvalue,'patients':len(delta)})
    ct=pd.DataFrame(contrasts);ct['BH_q']=bh(ct.paired_patient_signflip_p.to_numpy());ct.to_csv(T/'training_strategy_paired_comparisons.csv',index=False)
    # Freeze full-data deployment fit. Its training accuracy is never used as evidence.
    chosen_entry=models[fixed_id]
    candidates={i:[] for i in range(3)}
    for rec in chosen_entry['records']:
        for c in rec['candidates']:candidates[c['candidate']].append(c['inner_MSE'])
    ci=min(candidates,key=lambda k:np.mean(candidates[k]));cfg=tr.CONFIG['configs'][fixed.family][ci]
    if fixed.family in ['PCA','NMF','Coexpression']:
        selected=[rec['selected']['config']['rank'] for rec in chosen_entry['records']]
        rank=min(set(selected),key=lambda k:(-selected.count(k),k));cfg={'rank':rank,'alpha':.1}
    full=np.arange(len(data['y']))
    with threadpool_limits(limits=2):model=tr.fit_model(full,fixed.group,fixed.encoder,fixed.family,cfg)
    deploy=T/'bridge_model_checkpoints/frozen_main';deploy.mkdir(exist_ok=True)
    joblib.dump(model,deploy/'bridge_model.joblib',compress=3)
    nested_genes=summarygenes['NESTED_SELECTOR'].query("scope=='primary'").set_index('gene')
    fixed_genes=summarygenes[fixed_id].query("scope=='primary'").set_index('gene')
    supported=(nested_genes.stable_exploratory_candidate & fixed_genes.stable_exploratory_candidate &
        nested_genes.MSE_ratio_CI_high.lt(1)&fixed_genes.MSE_ratio_CI_high.lt(1))
    candidate=nested_genes.loc[supported].copy();candidate.to_csv(G/'supported_gene_candidates_for_downstream.csv')
    frozen={'frozen_at':datetime.now().astimezone().isoformat(),'model_id':fixed_id,
      'encoder':fixed.encoder,'group':fixed.group,'family':fixed.family,'configuration':cfg,
      'genes_QC_pass_full_fit':int(model.target.mask.sum()),'all_panel_RNA_genes':6175,
      'primary_nested_pipeline_MSE_ratio':ns['primary_MSE_ratio'],'primary_nested_pipeline_ratio_CI':[ns['ratio_CI_low'],ns['ratio_CI_high']],
      'fixed_family_primary_MSE_ratio':float(fixed.primary_MSE_ratio),'fixed_family_ratio_CI':[float(fixed.ratio_CI_low),float(fixed.ratio_CI_high)],
      'stable_nested_candidates':int(nested_genes.stable_exploratory_candidate.sum()),
      'supported_in_both_nested_and_fixed_with_error_CI':int(supported.sum()),'supported_genes':candidate.index.tolist(),
      'inference_unit':'Mean encoder features of multiple 112um physical patches; patch predictions are not measured cell expression',
      'deployment_status':'Exploratory frozen bridge; no independent molecular validation or clinical efficacy established',
      'clinical_overlap':'All 26 allowed in this stage; 22 overlap local ESD; downstream needs explicit overlap-aware validation',
      'LNM_AUC_used_for_selection':False,'full_fit_training_metrics_reported_as_validation':False,
      'main_checkpoint_sha256':tr.sha(deploy/'bridge_model.joblib'),'run':str(run)}
    tr.dump(deploy/'frozen_model_card.json',frozen)
    tr.dump(R/'07_reports/stage2_primary_results.json',frozen)
    print(json.dumps(frozen,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
