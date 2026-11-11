"""Exploratory sensitivity inference; explicitly keeps the primary result unchanged."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from evaluate_bridge import gene_metrics,bh
from full_embedding_ridge import FullRidge
from bridge_core import patient_weights
import train_bridge as tr

R=tr.R;T=tr.T;G=R/'04_gene_predictability'
def main():
    d=tr.load_data();tr.DATA=d
    out=T/'full_embedding_capacity_sensitivity/v2'
    assert json.loads((out/'completion.json').read_text())['status']=='COMPLETED'
    table=pd.read_csv(out/'full_embedding_Ridge_comparison.csv');tests=[];genes=[]
    for row in table.itertuples():
        a=np.load(out/f'{row.model_id}_OOF.npz');use=d['primary']
        frame,errors,baseline=gene_metrics(d['y'][use],a['prediction'][use],a['baseline'][use],
             d['patients'][use],d['fold'][use],d['genes'],a['gene_mask'],'primary',row.model_id,d['raw'][use])
        frame['analysis_status']='Exploratory capacity sensitivity; not the frozen main nested gene discovery'
        genes.append(frame)
        delta=baseline.mean(1)-errors.mean(1);rng=np.random.default_rng(20261104)
        signs=rng.choice([-1,1],size=(10000,len(delta)))
        tests.append((1+(signs@delta/len(delta)>=delta.mean()).sum())/(len(signs)+1))
    table['paired_patient_improvement_p']=tests;table['BH_q_six_exploratory_capacity_models']=bh(np.array(tests))
    table.to_csv(out/'full_embedding_Ridge_comparison.csv',index=False)
    pd.concat(genes,ignore_index=True).to_csv(G/'full_embedding_Ridge_gene_predictability_exploratory.csv.gz',index=False,compression='gzip')
    # Freeze a compact reference of the conservative primary-only full-feature ablation.
    name='ResNet50_FULL_A_Ridge'
    records=[json.loads((out/f'fold_{fold}/{name}/result.json').read_text(encoding='utf-8')) for fold in range(5)]
    selected=[r['selected_alpha_per_row'] for r in records]
    alpha=min(set(selected),key=lambda k:(-selected.count(k),-k))
    x=tr.feature_data('ResNet50','HSV_mean');use=np.flatnonzero(d['primary'])
    model=FullRidge().fit(x[use],d['y'][use],d['raw'][use],d['negative'][use],d['patients'][use],alpha)
    folder=T/'bridge_model_checkpoints/exploratory_full_embedding_Ridge';folder.mkdir(exist_ok=True)
    joblib.dump(model,folder/'model.joblib',compress=3)
    before=model.predict(x[:3]);loaded=joblib.load(folder/'model.joblib')
    assert np.array_equal(before,loaded.predict(x[:3]))
    row=table.loc[table.model_id.eq(name)].iloc[0]
    tr.dump(folder/'model_card.json',{'model_id':name,'alpha_per_training_row':alpha,
       'status':'Exploratory capacity sensitivity, not independent validation and not substituted for the primary model',
       'primary_MSE_ratio':float(row.primary_MSE_ratio),'CI':[float(row.ratio_CI_low),float(row.ratio_CI_high)],
       'BH_q_within_six_exploratory_capacity_models':float(row.BH_q_six_exploratory_capacity_models),
       'serialization_roundtrip_exact':True,'gene_specific_reliability_not_established':True})
    # Save the promising but multiplicity-nonsignificant NMF refinement only as an auxiliary reference.
    run=Path(json.loads((T/'active_run_all_cells_0.95_0.1.json').read_text(encoding='utf-8'))['run']);tr.RUN=run
    choices=[json.loads((run/f'outer_fold_{f}/ResNet50_HSV_mean_C_NMF/result.json').read_text(encoding='utf-8'))['selected']['config']['rank'] for f in range(5)]
    rank=min(set(choices),key=lambda k:(-choices.count(k),k))
    nmf=tr.fit_model(np.arange(len(d['y'])),'C','ResNet50','NMF',{'rank':rank,'alpha':.1})
    folder2=T/'bridge_model_checkpoints/auxiliary_NMF_C';folder2.mkdir(exist_ok=True)
    joblib.dump(nmf,folder2/'model.joblib',compress=3)
    tr.dump(folder2/'model_card.json',{'model_id':'ResNet50_HSV_mean_C_NMF','rank':rank,
       'status':'Secondary exploratory program reference; not reliable after 18-program-family multiplicity correction',
       'LNM_claims':False,'independent_validation':False})
    print(table[['model_id','primary_MSE_ratio','BH_q_six_exploratory_capacity_models']].to_string(index=False))
if __name__=='__main__':
    with threadpool_limits(limits=2):main()
