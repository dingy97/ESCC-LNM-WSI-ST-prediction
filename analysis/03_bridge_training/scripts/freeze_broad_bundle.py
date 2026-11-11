"""Honor the user's clarified whole-profile goal; reuse full-output fitted checkpoints."""
from pathlib import Path
from datetime import datetime
import json
import shutil
import joblib
import numpy as np
import pandas as pd
import train_bridge as tr
from full_embedding_ridge import FullRidge
R=tr.R;T=tr.T
def main():
    d=tr.load_data();tr.DATA=d;out=T/'bridge_model_checkpoints/broad_transcriptome';out.mkdir(exist_ok=True)
    configurations={}
    x=tr.feature_data('Phikon','HSV_mean')
    for name in ['Phikon_FULL_B_Ridge','Phikon_FULL_C_Ridge']:
        records=[json.loads((T/f'full_embedding_capacity_sensitivity/v2/fold_{f}/{name}/result.json').read_text(encoding='utf-8')) for f in range(5)]
        values=[row['selected_alpha_per_row'] for row in records]
        alpha=min(set(values),key=lambda a:(-values.count(a),-a));configurations[name]=alpha
        fitted=FullRidge().fit(x,d['y'],d['raw'],d['negative'],d['patients'],alpha,
                              d['primary'] if name=='Phikon_FULL_C_Ridge' else None)
        joblib.dump(fitted,out/(name+'.joblib'),compress=3)
    shutil.copy2(T/'bridge_model_checkpoints/auxiliary_NMF_C/model.joblib',out/'ResNet50_NMF_C.joblib')
    model=joblib.load(out/'Phikon_FULL_B_Ridge.joblib');program=joblib.load(out/'ResNet50_NMF_C.joblib')
    table=pd.read_csv(T/'full_embedding_capacity_sensitivity/v2/full_embedding_Ridge_comparison.csv')
    row=table.loc[table.model_id.eq('Phikon_FULL_B_Ridge')].iloc[0]
    card={'frozen_at':datetime.now().astimezone().isoformat(),
      'objective':'Broad multioutput H&E-to-RNA and molecular-program representation learning; not replication of a small-gene-panel study',
      'user_scope_clarification_verbatim':'我不是要像范围一样筛出来4个基因，我只是想借鉴机器学习的方式',
      'gene_names':d['genes'].tolist(),'RNA_gene_outputs':6175,'full_fit_learned_RNA_targets':int(model.target.mask.sum()),
      'image_QC_eligible_FOVs':375,'independent_patients':26,'primary_eval_patients':24,
      'main_broad_candidate':'Phikon full-embedding Ridge / Group B',
      'auxiliary_refinement':'Phikon full-embedding Ridge / Group C',
      'program_representation':'ResNet50 broad-then-primary NMF latent scores',
      'NMF_components':int(program.program.rank),
      'primary_frozen_analysis':'Original 42-model nested pipeline retained; MSE ratio 1.016945, CI includes 1',
      'broad_candidate_primary_MSE_ratio':float(row.primary_MSE_ratio),
      'broad_candidate_ratio_CI':[float(row.ratio_CI_low),float(row.ratio_CI_high)],
      'candidate_selection_basis':'Whole-panel held-primary-patient RNA prediction error, not gene whitelist and not LNM AUC',
      'analysis_status':'Full-embedding capacity family is exploratory and selection-exposed; not substituted for the frozen primary analysis',
      'no_four_gene_restriction':True,
      'gene_quality_use':'Per-gene predictability describes heterogeneous model accuracy; it does not predefine a four-gene input to downstream ML',
      'QC_fallback':'RNA genes failing full-training QC return their training means, clearly flagged; keeping 6175 outputs does not imply all genes are predictable',
      'inference_unit':'Physical approximately 512um FOV bags built from multiple native approximately 112um patches, frozen encoder feature pooling; not measured cell-level RNA',
      'downstream_plan':'Compare full inferred RNA, train-fold program embeddings, gene/program joint representations and pure-WSI features. Any further feature selection is trained within clinical development folds, never pre-restricted to four genes.',
      'clinical_overlap':'22 CosMx patients overlap 160 local ESD cases; retain mapping and run overlap-aware downstream validation/sensitivity',
      'TCGA_limit':'Previously tested cohort and Phikon TCGA pretraining; no claim of wholly unseen external evaluation',
      'model_configurations':configurations,
      'gene_diagnostic_or_shortlist_required_to_fit_or_predict':False,
      'alpha_interpretation':'Ridge penalty = stored alpha_per_training_row times number of training FOVs; patient weights normalized to mean 1',
      'program_configuration':{'rank':int(program.program.rank),'regression_alpha_per_training_row':program.config['alpha'],'primary_refinement_shrinkage':.5},
      'model_files':{name:tr.sha(out/name) for name in ['Phikon_FULL_B_Ridge.joblib','Phikon_FULL_C_Ridge.joblib','ResNet50_NMF_C.joblib']}}
    tr.dump(out/'bundle_card.json',card)
    pd.DataFrame({'gene':d['genes'],'prediction_status':np.where(model.target.mask,'learned_target','training_mean_fallback'),
        'output_included':True}).to_csv(out/'all_gene_output_status.csv',index=False)
    previous=T/'bridge_model_checkpoints/exploratory_four_gene_bundle/bundle_card.json'
    if previous.exists():
        old=json.loads(previous.read_text(encoding='utf-8'))
        old['main_delivery_role']='SUPERSEDED: historical individual-gene exploratory diagnostics only; no dependency of the broad model'
        tr.dump(previous,old)
    print('Frozen whole-transcriptome bundle: 6175 RNA outputs and '+str(program.program.rank)+' program scores; no four-gene restriction')
if __name__=='__main__':main()
