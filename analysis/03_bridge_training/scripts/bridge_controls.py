"""Composition crossfit and complete-patient permutation controls, not clinical tasks."""
import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from bridge_core import patient_weights,weighted_ridge,safe_correlations
from bridge_models import FeatureSpace,TargetSpace
import train_bridge as tr
from evaluate_bridge import bh,unit_arrays

R=tr.R;T=tr.T;O=R/'04_gene_predictability'

def composition_controls(data,run):
    selection=pd.read_csv(T/'nested_selection_by_fold.csv').set_index('outer_fold')
    counts=pd.read_csv(R/'00_inventory/fov_final_celltype_counts.csv').set_index('fov').reindex(data['FOV']).fillna(0)
    manifest=pd.read_csv(R/'fov_manifest.csv').set_index('FOV_ID').reindex(data['FOV'])
    qc=pd.read_csv(R/'02_expression/fov_expression_qc.csv').set_index('FOV_ID').reindex(data['FOV'])
    prop=(counts.to_numpy()+.5)/(counts.sum(1).to_numpy()[:,None]+.5*counts.shape[1])
    clr=np.log(prop)-np.log(prop).mean(1,keepdims=True)
    unknown=manifest.unannotated_cell_count.to_numpy()/manifest.raw_cell_count.to_numpy()
    numeric=np.column_stack([clr,unknown,np.log1p(qc.raw_cell_count),np.log1p(qc.RNA_total)])
    truth_res=np.zeros_like(data['y']);pred_res=np.zeros_like(data['y']);composition_pred=np.zeros_like(data['y'])
    audit=[]
    for fold in range(5):
        row=selection.loc[fold];folder=run/f'outer_fold_{fold}'/row.selected_model
        model=joblib.load(folder/'model.joblib')
        train=np.flatnonzero(data['fold']!=fold);test=np.flatnonzero(data['fold']==fold)
        if row.group=='A':train=train[data['primary'][train]]
        assert not set(data['patients'][train])&set(data['patients'][test])
        tissue=pd.get_dummies(data['m'].tissue_category).to_numpy(dtype=float)
        # Test-only categories are dropped using training variance, never learned from test targets.
        c=np.column_stack([numeric,tissue]);w=patient_weights(data['patients'][train])
        mean=np.average(c[train],axis=0,weights=w);sd=np.sqrt(np.average((c[train]-mean)**2,axis=0,weights=w))
        active=sd>1e-9;cc=(c[:,active]-mean[active])/sd[active]
        yy=data['y'][train];ym=np.average(yy,axis=0,weights=w)
        x=tr.feature_data(row.encoder,'HSV_mean');py=model.predict(x[train]);pm=np.average(py,axis=0,weights=w)
        cy=weighted_ridge(cc[train],yy-ym,w,len(train)*.1)
        cp=weighted_ridge(cc[train],py-pm,w,len(train)*.1)
        composition_pred[test]=ym+cc[test]@cy
        truth_res[test]=data['y'][test]-composition_pred[test]
        pred_res[test]=model.predict(x[test])-(pm+cc[test]@cp)
        audit.append({'outer_fold':fold,'training_patients':sorted(set(data['patients'][train])),
          'heldout_patients':sorted(set(data['patients'][test])),'covariates':
          'seven annotated-cell CLR proportions, unannotated fraction, log cells, log RNA depth, training-observed specimen categories',
          'separate_truth_and_prediction_residualizers':True})
    np.savez_compressed(T/'composition_crossfit_predictions.npz',FOV_ID=data['FOV'],genes=data['genes'],
                       true_residual=truth_res,predicted_residual=pred_res,composition_prediction=composition_pred)
    rows=[]
    for scope,use in [('primary',data['primary']),('broad',np.ones(len(data['y']),bool))]:
        units=np.unique(data['patients'][use])
        y=np.stack([truth_res[use][data['patients'][use]==u].mean(0) for u in units])
        p=np.stack([pred_res[use][data['patients'][use]==u].mean(0) for u in units])
        rows.append(pd.DataFrame({'gene':data['genes'],'scope':scope,'patients':len(units),
            'composition_adjusted_patient_Pearson':safe_correlations(y,p),
            'composition_adjusted_patient_Spearman':safe_correlations(rankdata(y,axis=0),rankdata(p,axis=0))}))
    pd.concat(rows).to_csv(O/'composition_adjusted_gene_predictability.csv',index=False)
    tr.dump(T/'composition_control_audit.json',{'fits':audit,
       'limit':'Measured composition/depth are diagnostic covariates unavailable to WSI-only deployment; linear adjustment is not causal separation'})
    print('Patient-crossfit composition controls completed',flush=True)

def permutations(data,repeats):
    m=data['m'];use=data['primary'];patients=np.unique(data['patients'][use])
    x=tr.feature_data('ResNet50','HSV_mean')
    xx=np.stack([x[use][data['patients'][use]==u].mean(0) for u in patients])
    yy=np.stack([data['y'][use][data['patients'][use]==u].mean(0) for u in patients])
    raw=np.stack([data['raw'][use][data['patients'][use]==u].mean(0) for u in patients])
    neg=np.stack([data['negative'][use][data['patients'][use]==u].mean(0) for u in patients])
    folds=np.array([data['fold'][np.flatnonzero(data['patients']==u)[0]] for u in patients])
    prep={}
    for fold in range(5):
        train=folds!=fold;test=~train;feature=FeatureSpace(xx[train],patients[train])
        prep[fold]=(train,test,feature.transform(xx[train]),feature.transform(xx[test]))
    mapping=[];all_stats=[];total_stats=[];observed_prediction=None;observed_baseline=None
    memory=np.lib.format.open_memmap(T/'patient_permutation_OOF_predictions.npy',mode='w+',dtype='float32',
                shape=(repeats+1,len(patients),len(data['genes'])))
    rng=np.random.default_rng(20261104)
    for repeat in range(repeats+1):
        order=np.arange(len(patients)) if repeat==0 else rng.permutation(len(patients))
        target=yy[order];prediction=np.zeros_like(target);baseline=np.zeros_like(target)
        for fold in range(5):
            train,test,a,b=prep[fold]
            ts=TargetSpace(target[train],raw[order][train],neg[order][train],patients[train])
            coef=weighted_ridge(a,ts.transform(target[train]),np.ones(train.sum()),.1*train.sum())
            prediction[test]=np.clip(ts.decode(b@coef),0,np.log1p(1e6));baseline[test]=ts.mean
        error=((target-prediction)**2).mean(0);be=((target-baseline)**2).mean(0)
        memory[repeat]=prediction.astype('float32')
        all_stats.append(error/np.maximum(be,1e-12));total_stats.append(float(error.mean()/be.mean()));mapping.append(order)
        if repeat==0:observed_prediction=prediction.copy();observed_baseline=baseline.copy()
        if repeat%100==0:print(f'Complete-patient refit permutation {repeat}/{repeats}',flush=True)
    memory.flush();stats=np.stack(all_stats)
    observed=stats[0];pvalue=(1+(stats[1:]<=observed).sum(0))/(repeats+1)
    pd.DataFrame({'gene':data['genes'],'observed_patient_level_MSE_ratio':observed,
        'patient_permutation_p':pvalue,'patient_permutation_BH_q':bh(pvalue)}).to_csv(O/'patient_aggregate_Ridge_permutation_gene_control.csv',index=False)
    np.savez_compressed(T/'patient_permutation_statistics.npz',MSE_ratio=stats,
         donor_permutation=np.array(mapping),Patient_ID=patients,genes=data['genes'],fold=folds,
         observed_prediction=observed_prediction,observed_baseline=observed_baseline)
    # Recompute the actual whole-panel endpoint, not a mean of unstable per-gene ratios.
    observed_total=np.mean((yy-observed_prediction)**2)/np.mean((yy-observed_baseline)**2)
    null_scalar=np.asarray(total_stats[1:]);observed_scalar=total_stats[0]
    pd.DataFrame({'permutation_id':np.arange(repeats+1),'whole_panel_MSE_ratio':total_stats}).to_csv(T/'patient_permutation_whole_panel_results.csv',index=False)
    tr.dump(T/'patient_permutation_control_summary.json',{
       'patients':len(patients),'permutations':repeats,'observed_aggregate_MSE_ratio':float(observed_total),
       'null_whole_panel_MSE_ratio_median':float(np.median(null_scalar)),
       'permutation_p':float((1+(null_scalar<=observed_scalar).sum())/(repeats+1)),
       'protocol':'Refit fixed ResNet50/Ridge on 24 patient-aggregated primary image/RNA pairs; permute full RNA/control profiles as patient units and repeat every fold-specific target transform',
       'scope_limit':'Auxiliary patient-level refit null, not an exact FOV-level or complete adaptive-model-selection permutation test; no LNM labels involved'})
    print('Whole-patient refit negative control completed',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--permutations',type=int,default=1000);args=parser.parse_args()
    data=tr.load_data();tr.DATA=data
    active=json.loads((T/'active_run_all_cells_0.95_0.1.json').read_text(encoding='utf-8'));run=Path(active['run']);tr.RUN=run
    composition_controls(data,run);permutations(data,args.permutations)
if __name__=='__main__':
    with threadpool_limits(limits=2):main()
