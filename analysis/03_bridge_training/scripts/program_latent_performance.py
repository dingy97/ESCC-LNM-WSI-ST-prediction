"""Prespecified secondary program-score endpoint; no pooling of component identities."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import train_bridge as tr
from evaluate_bridge import ci_ratio,bh

R=tr.R;T=tr.T;O=R/'05_molecular_programs'
def main():
    data=tr.load_data();run=Path(json.loads((T/'active_run_all_cells_0.95_0.1.json').read_text(encoding='utf-8'))['run'])
    rows=[];details=[]
    for encoder in tr.CONFIG['encoders']:
        for group in ['A','B','C']:
            for family in ['PCA','NMF','Coexpression']:
                name=f'{encoder}_HSV_mean_{group}_{family}';errors=[];baselines=[];fold_improvements=[]
                for fold in range(5):
                    folder=run/f'outer_fold_{fold}'/name
                    model=joblib.load(folder/'model.joblib');a=np.load(folder/'program_loadings_and_scores.npz')
                    pos=pd.Index(data['FOV']).get_indexer(a['test_FOV']);keep=data['primary'][pos]
                    y=(a['test_true_scores'][keep]-model.program.mean)/model.program.scale
                    p=(a['test_predicted_scores'][keep]-model.program.mean)/model.program.scale
                    groups=data['patients'][pos][keep]
                    fe=[];fb=[]
                    for u in np.unique(groups):
                        ix=groups==u;e=float(((y[ix]-p[ix])**2).mean());b=float((y[ix]**2).mean())
                        errors.append(e);baselines.append(b);fe.append(e);fb.append(b)
                        details.append({'model_id':name,'outer_fold':fold,'Patient_ID':u,
                            'training_standardized_latent_MSE':e,'latent_baseline_MSE':b,'rank':y.shape[1]})
                    fold_improvements.append(np.mean(fe)<np.mean(fb))
                errors=np.array(errors);baseline=np.array(baselines);ci=ci_ratio(errors,baseline)
                delta=baseline-errors;rng=np.random.default_rng(20261104);sign=rng.choice([-1,1],size=(10000,len(delta)))
                pvalue=(1+(sign@delta/len(delta)>=delta.mean()).sum())/(len(sign)+1)
                rows.append({'model_id':name,'encoder':encoder,'group':group,'family':family,
                   'standardized_program_MSE_ratio':errors.mean()/baseline.mean(),'ratio_CI_low':ci[0],'ratio_CI_high':ci[1],
                   'patient_paired_improvement_signflip_p':pvalue,'positive_folds':sum(fold_improvements),'primary_patients':len(errors),
                   'metric_scope':'Equal-patient, mean training-standardized latent component loss; target basis/rank is training-fold specific',
                   'selection_status':'Secondary program endpoint; full-panel gene primary endpoint unchanged'})
    table=pd.DataFrame(rows);table['program_family_BH_q']=bh(table.patient_paired_improvement_signflip_p.to_numpy())
    table['program_family_candidate']=table.ratio_CI_high.lt(1)&table.program_family_BH_q.lt(.05)&table.positive_folds.ge(4)
    table.to_csv(O/'program_model_comparison.csv',index=False)
    pd.DataFrame(details).to_csv(O/'program_per_patient_prediction_errors.csv',index=False)
    print(table.sort_values('standardized_program_MSE_ratio')[['model_id','standardized_program_MSE_ratio','ratio_CI_low','ratio_CI_high','program_family_BH_q','program_family_candidate']].head(8).to_string(index=False))
if __name__=='__main__':
    with threadpool_limits(limits=2):main()
