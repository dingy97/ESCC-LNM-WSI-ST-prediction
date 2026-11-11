"""Whole-transcriptome OOF and profile diagnostics; no small gene panel gate."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from bridge_core import safe_correlations
from full_embedding_ridge import FullRidge
import train_bridge as tr
R=tr.R;T=tr.T
def rowcor(a,b):return safe_correlations(a.T,b.T)
def main():
    data=tr.load_data();tr.DATA=data;rows=[]
    for name,prefix in [('Phikon_FULL_B_Ridge','broad_transcriptome_oof_predictions'),('Phikon_FULL_C_Ridge','primary_refinement_oof_predictions')]:
        a=np.load(T/f'full_embedding_capacity_sensitivity/v2/{name}_OOF.npz')
        pred=a['prediction'];base=a['baseline']
        frame=pd.DataFrame(pred,columns=data['genes'])
        for col in ['outer_fold','Patient_ID','FOV_key','FOV_ID']:frame.insert(0,col,data['m'][col].to_numpy())
        frame.to_csv(T/(prefix+'.csv.gz'),index=False,compression='gzip')
        for scope,use in [('primary',data['primary']),('broad',np.ones(len(data['y']),bool))]:
            patients=np.unique(data['patients'][use])
            y=np.stack([data['y'][use][data['patients'][use]==u].mean(0) for u in patients])
            p=np.stack([pred[use][data['patients'][use]==u].mean(0) for u in patients])
            b=np.stack([base[use][data['patients'][use]==u].mean(0) for u in patients])
            corr=rowcor(y,p);bcorr=rowcor(y,b)
            rho=rowcor(rankdata(y,axis=1),rankdata(p,axis=1));brho=rowcor(rankdata(y,axis=1),rankdata(b,axis=1))
            for i,u in enumerate(patients):rows.append({'model_id':name,'scope':scope,'Patient_ID':u,
                'profile_Pearson_across_genes':corr[i],'baseline_profile_Pearson':bcorr[i],
                'profile_Spearman_across_genes':rho[i],'baseline_profile_Spearman':brho[i],
                'RNA_targets':6175,'interpretation':'Profile correlation can be high from between-gene mean differences; always compare baseline and held-patient error. Genes are not independent patients.'})
    pd.DataFrame(rows).to_csv(T/'whole_transcriptome_profile_diagnostics.csv',index=False)
    # Actual fold-specific OOF program scores retain their training-basis identities.
    run=Path(json.loads((T/'active_run_all_cells_0.95_0.1.json').read_text(encoding='utf-8'))['run'])
    program=[]
    for fold in range(5):
        folder=run/f'outer_fold_{fold}/ResNet50_HSV_mean_C_NMF';a=np.load(folder/'program_loadings_and_scores.npz')
        model=joblib.load(folder/'model.joblib')
        for j in range(a['test_true_scores'].shape[1]):
            for i,fid in enumerate(a['test_FOV']):
                program.append({'FOV_ID':int(fid),'outer_fold':fold,'basis_ID':f'outer_fold_{fold}_training_NMF_C',
                    'component':j+1,'RNA_derived_score':float(a['test_true_scores'][i,j]),
                    'image_inferred_score':float(a['test_predicted_scores'][i,j]),
                    'RNA_derived_standardized_score':float((a['test_true_scores'][i,j]-model.program.mean[j])/model.program.scale[j]),
                    'image_inferred_standardized_score':float((a['test_predicted_scores'][i,j]-model.program.mean[j])/model.program.scale[j]),
                    'identity_note':'Do not concatenate different fold component IDs as the same biological program'})
    pd.DataFrame(program).to_csv(T/'program_oof_scores.csv',index=False)
    broadcard=json.loads((T/'bridge_model_checkpoints/broad_transcriptome/bundle_card.json').read_text(encoding='utf-8'))
    tr.dump(R/'07_reports/broad_transcriptome_delivery_summary.json',{k:v for k,v in broadcard.items() if k!='gene_names'})
    print('Whole-panel broad/refined OOF matrices and fold-specific program scores exported')
if __name__=='__main__':
    with threadpool_limits(limits=2):main()
