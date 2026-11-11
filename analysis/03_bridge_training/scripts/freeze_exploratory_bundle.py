"""Model-specific candidate routing, clearly separated from the frozen primary analysis."""
from pathlib import Path
from datetime import datetime
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import train_bridge as tr
from full_embedding_ridge import FullRidge

def main():
    R=tr.R;T=tr.T;data=tr.load_data();tr.DATA=data
    candidates=pd.read_csv(R/'04_gene_predictability/composition_robust_exploratory_gene_candidates.csv')
    rna=pd.read_csv(R/'04_gene_predictability/full_embedding_RNA_stratum_gene_metrics.csv.gz')
    audit=[]
    for row in candidates.itertuples():
        for version in ['annotated','unannotated']:
            result=rna.loc[rna.model_id.eq(row.model_id)&rna.gene.eq(row.gene)&rna.RNA_stratum.eq(version)].iloc[0]
            audit.append({'model_id':row.model_id,'gene':row.gene,'RNA_stratum':version,
                'Spearman_patient_mean':result.Spearman_patient_mean,'Spearman_patient_BH_q':result.Spearman_patient_BH_q,
                'MSE_ratio_training_mean':result.MSE_ratio_training_mean,
                'positive_patient_Spearman_folds':result.positive_patient_Spearman_folds,
                'interpretation':'Matched stratum sensitivity, no claim that annotation status equals quality'})
    pd.DataFrame(audit).to_csv(R/'04_gene_predictability/exploratory_gene_RNA_annotation_sensitivity.csv',index=False)
    folder=T/'bridge_model_checkpoints/exploratory_four_gene_bundle';folder.mkdir(exist_ok=True)
    route=[];models={}
    x=tr.feature_data('Phikon','HSV_mean')
    for model_id in sorted(candidates.model_id.unique()):
        group=model_id.split('_')[2]
        records=[json.loads((T/f'full_embedding_capacity_sensitivity/v2/fold_{f}/{model_id}/result.json').read_text(encoding='utf-8')) for f in range(5)]
        values=[r['selected_alpha_per_row'] for r in records]
        alpha=min(set(values),key=lambda a:(-values.count(a),-a))
        fitted=FullRidge().fit(x,data['y'],data['raw'],data['negative'],data['patients'],alpha,
                              data['primary'] if group=='C' else None)
        dest=folder/(model_id+'.joblib');joblib.dump(fitted,dest,compress=3)
        loaded=joblib.load(dest);assert np.array_equal(fitted.predict(x[:3]),loaded.predict(x[:3]))
        models[model_id]={'path':str(dest),'sha256':tr.sha(dest),'alpha_per_training_row':alpha,
            'training_FOVs':375,'patients':26,'all_gene_outputs_reliable':False}
        for gene in candidates.loc[candidates.model_id.eq(model_id),'gene']:
            route.append({'gene':gene,'model_id':model_id,'gene_output_index':int(np.flatnonzero(data['genes']==gene)[0]),
                'status':'EXPLORATORY_ONLY_NOT_INDEPENDENTLY_VALIDATED','LNM_association_established':False})
    pd.DataFrame(route).to_csv(folder/'gene_model_routing.csv',index=False)
    tr.dump(folder/'bundle_card.json',{'frozen_at':datetime.now().astimezone().isoformat(),
       'status':'Exploratory follow-up candidate bundle, not substituted for the negative frozen primary nested result',
       'genes':sorted(candidates.gene.unique().tolist()),'models':models,
       'eligibility_basis':'Six-model/6175-gene FDR, patient-bootstrap error interval and additional crossfit linear composition sensitivity on the same discovery cohort',
       'scope_limit':'No independent molecular validation; conditional OOF and analyst-selected capacity family; four genes are not established LNM markers',
       'input':'Phikon CLS mean of retained native approximately 112um patches within virtual approximately 512um FOVs; same physical scale and hard-mask pooling as training',
       'output':'image-inferred FOV expression only; other 6171 gene outputs have no demonstrated reliability',
       'patient_overlap':'22/26 overlap local ESD; keep overlap-aware downstream nested/crossfit sensitivities',
       'pretraining':'Phikon includes TCGA pretraining; later TCGA evaluation is not wholly unseen pretraining validation'})
    print('Frozen exploratory routing for four genes; primary negative reference unchanged')
if __name__=='__main__':
    with threadpool_limits(limits=2):main()
