from pathlib import Path
import json
import numpy as np
import pandas as pd
import train_bridge as tr
from evaluate_bridge import model_summary,gene_metrics
from bridge_core import patient_weights
from threadpoolctl import threadpool_limits

R=tr.R;T=tr.T;G=R/'04_gene_predictability'

def collect(run,name,data):
    pred=np.zeros_like(data['y'],dtype='float32');baseline=pred.copy();mask=np.ones(6175,bool);recs=[]
    for fold in range(5):
        folder=run/f'outer_fold_{fold}'/name
        rec=json.loads((folder/'result.json').read_text(encoding='utf-8'));a=np.load(folder/'OOF_prediction.npz')
        pos=pd.Index(data['FOV']).get_indexer(a['FOV_ID']);pred[pos]=a['prediction'];baseline[pos]=a['primary_training_mean']
        mask&=a['gene_mask'];recs.append(rec)
    return model_summary(name,{'pred':pred,'mask':mask,'records':recs},data,baseline),pred,baseline

def correct_secondary_broad_baseline(data):
    broad=np.zeros_like(data['y'],dtype='float64');tissue_base=broad.copy()
    for fold in range(5):
        train=np.flatnonzero(data['fold']!=fold);test=np.flatnonzero(data['fold']==fold)
        mean=np.average(data['y'][train],axis=0,weights=patient_weights(data['patients'][train]));broad[test]=mean
        for tissue in np.unique(data['m'].tissue_category):
            a=train[data['m'].tissue_category.to_numpy()[train]==tissue]
            b=test[data['m'].tissue_category.to_numpy()[test]==tissue]
            tissue_base[b]=np.average(data['y'][a],axis=0,weights=patient_weights(data['patients'][a])) if len(a) else mean
    np.savez_compressed(T/'secondary_broad_and_tissue_baselines.npz',FOV_ID=data['FOV'],broad_training_mean=broad,tissue_training_mean=tissue_base)
    for npz,filename in [('main_nested_OOF.npz','gene_predictability.csv'),('fixed_family_OOF.npz','fixed_deployment_gene_predictability.csv')]:
        a=np.load(T/npz);old=pd.read_csv(G/filename)
        original=G/(filename.replace('.csv','_prior_broad_primary_reference.csv'))
        if not original.exists():old.loc[old.scope.eq('broad')].to_csv(original,index=False)
        new,_,_=gene_metrics(data['y'],a['prediction'],broad,data['patients'],data['fold'],data['genes'],a['gene_mask'],'broad',
                         old.model_id.iloc[0],data['raw'])
        new['baseline_reference']='Broad training-patient mean'
        primary=old.loc[old.scope.eq('primary')].copy();primary['baseline_reference']='Primary training-patient mean'
        pd.concat([primary,new],ignore_index=True).to_csv(G/filename,index=False)
    # Update the descriptive by-patient broad baseline column consistently.
    by=pd.read_csv(G/'gene_predictability_by_patient.csv')
    for u in np.unique(data['patients']):
        values=np.sqrt(((data['y'][data['patients']==u]-broad[data['patients']==u])**2).mean(0))
        lookup=dict(zip(data['genes'],values));selected=by.Patient_ID.eq(u)&by.scope.eq('broad')
        by.loc[selected,'baseline_RMSE']=by.loc[selected,'gene'].map(lookup)
    by.to_csv(G/'gene_predictability_by_patient.csv',index=False)
    rows=[];a=np.load(T/'main_nested_OOF.npz');p=a['prediction']
    for tissue in np.unique(data['m'].tissue_category):
        use=data['m'].tissue_category.eq(tissue).to_numpy();units=np.unique(data['patients'][use])
        errors=np.array([((data['y'][use][data['patients'][use]==u]-p[use][data['patients'][use]==u])**2).mean() for u in units])
        base=np.array([((data['y'][use][data['patients'][use]==u]-tissue_base[use][data['patients'][use]==u])**2).mean() for u in units])
        rows.append({'tissue_category':tissue,'patients':len(units),'FOVs':int(use.sum()),
            'patient_MSE':errors.mean(),'MSE_ratio_tissue_training_mean':errors.mean()/base.mean(),
            'interpretation':'descriptive_only_small_n' if len(units)<5 else 'patient_crossfit_diagnostic'})
    pd.DataFrame(rows).to_csv(T/'tissue_stratified_prediction_performance.csv',index=False)
    tr.dump(T/'secondary_baseline_note.json',{'primary_endpoint_unchanged':True,
        'correction':'Broad secondary gene tables use broad-training means. Tissue diagnostics also include training-tissue means. Prior primary-mean broad tables retained as intermediates.',
        'no_model_retrained_or_reselected':True})

def main():
    data=tr.load_data();tr.DATA=data
    mainrun=Path(json.loads((T/'active_run_all_cells_0.95_0.1.json').read_text(encoding='utf-8'))['run'])
    card=json.loads((R/'07_reports/stage2_primary_results.json').read_text(encoding='utf-8'))
    main_id=card['model_id'];masks=[]
    for e in tr.CONFIG['encoders']:
        for ag in ['HSV_mean','all16_mean','OD_soft_mean']:
            for group in ['A','B','C']:
                name=f'{e}_{ag}_{group}_Ridge';row,_,_=collect(mainrun,name,data);row['aggregation']=ag;masks.append(row)
    pd.DataFrame(masks).to_csv(T/'image_mask_aggregation_sensitivity.csv',index=False)
    qc=[];rna=[]
    for file in sorted(T.glob('active_run_*.json')):
        record=json.loads(file.read_text(encoding='utf-8'));ident=record['identity'];run=Path(record['run'])
        if ident['RNA']=='all_cells':
            name=main_id
            if not all((run/f'outer_fold_{f}/{name}/result.json').exists() for f in range(5)):continue
            row,_,_=collect(run,name,data);row.update(negative_quantile=ident['negative_quantile'],patient_support=ident['patient_support'])
            counts=[json.loads((run/f'outer_fold_{f}/{name}/result.json').read_text(encoding='utf-8'))['QC_genes_passed'] for f in range(5)]
            row.update(QC_genes_outer_min=min(counts),QC_genes_outer_max=max(counts));qc.append(row)
        else:
            sensitive=tr.load_data(ident['RNA'])
            for name in [main_id,'ResNet50_HSV_mean_B_MLP']:
                row,_,_=collect(run,name,sensitive);row['RNA_stratum']=ident['RNA'];rna.append(row)
    for name in [main_id,'ResNet50_HSV_mean_B_MLP']:
        row,_,_=collect(mainrun,name,data);row['RNA_stratum']='all_cells';rna.append(row)
    pd.DataFrame(qc).sort_values(['negative_quantile','patient_support']).to_csv(T/'gene_QC_threshold_sensitivity.csv',index=False)
    pd.DataFrame(rna).to_csv(T/'RNA_annotation_sensitivity.csv',index=False)
    correct_secondary_broad_baseline(data)
    print('18 mask/aggregation, 9 gene-QC, 6 RNA-stratum comparisons summarized; primary endpoint unchanged')
if __name__=='__main__':
    with threadpool_limits(limits=2):main()
