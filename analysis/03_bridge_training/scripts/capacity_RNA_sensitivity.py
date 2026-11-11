"""Matched exploratory full-embedding RNA-stratum refits; inherit only training-chosen alpha."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from full_embedding_ridge import FullRidge
from bridge_core import patient_weights,patient_mse
from evaluate_bridge import gene_metrics,model_summary
import train_bridge as tr
R=tr.R;T=tr.T;O=T/'full_embedding_RNA_sensitivity';O.mkdir(exist_ok=True)
def main():
    rows=[];gene_rows=[]
    for version in ['annotated','unannotated']:
        data=tr.load_data(version);tr.DATA=data
        for group in ['B','C']:
            name=f'Phikon_FULL_{group}_Ridge';x=tr.feature_data('Phikon','HSV_mean')
            prediction=np.zeros_like(data['y'],dtype='float32');baseline=prediction.copy();mask=np.ones(6175,bool);records=[]
            for fold in range(5):
                source=T/f'full_embedding_capacity_sensitivity/v2/fold_{fold}/{name}/result.json'
                alpha=json.loads(source.read_text(encoding='utf-8'))['selected_alpha_per_row']
                train=np.flatnonzero(data['fold']!=fold);test=np.flatnonzero(data['fold']==fold)
                model=FullRidge().fit(x[train],data['y'][train],data['raw'][train],data['negative'][train],data['patients'][train],alpha,
                                      data['primary'][train] if group=='C' else None)
                prediction[test]=model.predict(x[test]);mask&=model.target.mask
                p=train[data['primary'][train]];baseline[test]=np.average(data['y'][p],axis=0,weights=patient_weights(data['patients'][p]))
                folder=O/f'{version}/fold_{fold}/{name}';folder.mkdir(parents=True,exist_ok=True)
                joblib.dump(model,folder/'model.joblib',compress=3)
                np.savez_compressed(folder/'OOF_prediction.npz',FOV_ID=data['FOV'][test],prediction=prediction[test],baseline=baseline[test],gene_mask=model.target.mask)
                held_primary=test[data['primary'][test]]
                ratio=patient_mse(data['y'][held_primary],prediction[held_primary],data['patients'][held_primary])/patient_mse(data['y'][held_primary],baseline[held_primary],data['patients'][held_primary])
                records.append({'encoder':'Phikon','family':'Full_embedding_Ridge','group':group,'solver_converged':True,'primary_MSE_ratio':ratio})
                tr.dump(folder/'training_provenance.json',{'alpha':alpha,'alpha_source':'all-cell inner patient CV from same outer training patients only',
                  'RNA_version':version,'training_patients':sorted(set(data['patients'][train])),
                  'heldout_patients':sorted(set(data['patients'][test])),'all_stratum_transforms_training_only':True})
            entry={'pred':prediction,'mask':mask,'records':records};summary=model_summary(name,entry,data,baseline)
            # The summary's fold-improvement count needs the actual newly evaluated fold scores.
            summary.pop('positive_improvement_folds');summary['RNA_stratum']=version;rows.append(summary)
            use=data['primary'];metrics,_,_=gene_metrics(data['y'][use],prediction[use],baseline[use],data['patients'][use],data['fold'][use],data['genes'],mask,'primary',name,data['raw'][use])
            metrics['RNA_stratum']=version;gene_rows.append(metrics)
            np.savez_compressed(O/f'{version}_{name}_OOF.npz',FOV_ID=data['FOV'],genes=data['genes'],prediction=prediction,baseline=baseline,gene_mask=mask)
            print(f'{version} {name} RNA sensitivity complete',flush=True)
    pd.DataFrame(rows).to_csv(O/'full_embedding_RNA_stratum_comparison.csv',index=False)
    pd.concat(gene_rows).to_csv(R/'04_gene_predictability/full_embedding_RNA_stratum_gene_metrics.csv.gz',index=False,compression='gzip')
    tr.dump(O/'completion.json',{'status':'COMPLETED','outer_refits':20,
      'selection_limit':'Exploratory capacity follow-up; primary results unchanged; inherited alpha is training-only, no held-out RNA retuning'})
if __name__=='__main__':
    with threadpool_limits(limits=2):main()
