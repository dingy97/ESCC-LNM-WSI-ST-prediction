"""Fold-specific program evaluation plus explicitly labelled full-training reference programs."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata,hypergeom
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
from bridge_core import safe_correlations
from bridge_models import TargetSpace,ProgramSpace
import train_bridge as tr
from evaluate_bridge import bh,choose_fixed

R=tr.R;T=tr.T;O=R/'05_molecular_programs'

def summarize_fold_programs(data,run):
    rows=[]
    for path in sorted(run.glob('outer_fold_*/*/program_loadings_and_scores.npz')):
        rec=json.loads((path.parent/'result.json').read_text(encoding='utf-8'))
        if rec['aggregation']!='HSV_mean':continue
        a=np.load(path);pos=pd.Index(data['FOV']).get_indexer(a['test_FOV'])
        for scope,keep in [('primary',data['primary'][pos]),('broad',np.ones(len(pos),bool))]:
            groups=data['patients'][pos][keep];units=np.unique(groups)
            y=a['test_true_scores'][keep];p=a['test_predicted_scores'][keep]
            if len(units)<3:continue
            yy=np.stack([y[groups==u].mean(0) for u in units]);pp=np.stack([p[groups==u].mean(0) for u in units])
            cor=safe_correlations(rankdata(yy,axis=0),rankdata(pp,axis=0))
            meanerrs=np.stack([((y-p)[groups==u]**2).mean(0) for u in units]).mean(0)
            base=joblib.load(path.parent/'model.joblib').program.mean
            baseerrs=np.stack([((y-base)[groups==u]**2).mean(0) for u in units]).mean(0)
            for k in range(y.shape[1]):
                rows.append({'model_id':rec['matrix_id'],'encoder':rec['encoder'],'group':rec['group'],
                    'family':rec['family'],'outer_fold':rec['outer_fold'],'scope':scope,
                    'program_local_ID':f'{rec["family"]}_fold{rec["outer_fold"]}_component{k+1}',
                    'rank':y.shape[1],'patient_Spearman':cor[k],
                    'RMSE_patient_balanced':np.sqrt(meanerrs[k]),'MSE_ratio_program_training_mean':meanerrs[k]/max(baseerrs[k],1e-12),
                    'patients':len(units),'FOVs':len(y),'inner_loading_stability':rec['selected']['loading_stability'],
                    'identity_limit':'Fold-specific basis; do not pool different fold component IDs as one fixed global program'})
    pd.DataFrame(rows).to_csv(O/'program_predictability.csv',index=False)

def gene_query(loadings,sign):
    value=np.maximum(loadings if sign=='positive' else -loadings,0)
    if np.sum(value**2)==0:return np.array([],int)
    order=np.argsort(-value**2)
    n=np.searchsorted(np.cumsum(value[order]**2),.5*np.sum(value**2))+1
    return order[:n]

def reference_programs(data,run):
    table=pd.read_csv(T/'bridge_model_comparison.csv')
    full=np.arange(len(data['y']));all_enrich=[];reference=[]
    resource=next((T/'config/public_resources').glob('GO_Biological_Process_*.gmt'),None)
    terms={}
    if resource:
        for line in resource.read_text(encoding='utf-8').splitlines():
            fields=line.split('\t')
            if len(fields)>=3:terms[fields[0]]=set(x.upper() for x in fields[2:] if x)
    measured=set(data['genes']);annotated=set().union(*terms.values()) if terms else set()
    universe=measured&annotated
    for family in ['PCA','NMF','Coexpression']:
        choice=choose_fixed(table.loc[table.family.eq(family)])
        folders=list(run.glob(f'outer_fold_*/{choice.model_id}/result.json'))
        selected=[json.loads(p.read_text(encoding='utf-8'))['selected']['config']['rank'] for p in folders]
        rank=min(set(selected),key=lambda k:(-selected.count(k),k))
        model=tr.fit_model(full,choice.group,choice.encoder,family,{'rank':rank,'alpha':.1})
        folder=O/f'{family}_training_reference';folder.mkdir(exist_ok=True)
        joblib.dump(model,folder/'reference_model.joblib',compress=3)
        loading=model.program.full_loadings()
        pd.DataFrame(loading.T,index=data['genes'],columns=[f'{family}_{k+1}' for k in range(rank)]).rename_axis('gene').to_csv(folder/'gene_loadings.csv.gz',compression='gzip')
        actual=model.program.true_scores(data['y'])
        predicted=model.latent_predict(model.feature.transform(tr.feature_data(choice.encoder,'HSV_mean')))*model.program.scale+model.program.mean
        for kind,value in [('actual_training_reference_scores',actual),('image_training_reference_scores',predicted)]:
            frame=pd.DataFrame(value,columns=[f'{family}_{k+1}' for k in range(rank)])
            frame.insert(0,'FOV_ID',data['FOV']);frame['evaluation_status']='Full-training descriptive reference; not OOF validation'
            frame.to_csv(folder/f'{kind}.csv',index=False)
        actual_frame=pd.DataFrame(actual,columns=[f'{family}_{k+1}' for k in range(rank)])
        actual_frame['tissue_category']=data['m'].tissue_category.to_numpy()
        actual_frame.groupby('tissue_category').mean().to_csv(folder/'tissue_program_means_descriptive.csv')
        comp=pd.read_csv(R/'00_inventory/fov_final_celltype_counts.csv').set_index('fov').reindex(data['FOV']).fillna(0)
        props=(comp.to_numpy()+.5)/(comp.sum(1).to_numpy()[:,None]+.5*comp.shape[1])
        correlations=[]
        for k in range(rank):
            for j,cell in enumerate(comp.columns):
                yy=np.stack([actual[data['patients']==u,k].mean() for u in np.unique(data['patients'])])[:,None]
                pp=np.stack([props[data['patients']==u,j].mean() for u in np.unique(data['patients'])])[:,None]
                correlations.append({'program':f'{family}_{k+1}','celltype':cell,
                    'patient_Spearman_descriptive':safe_correlations(rankdata(yy,axis=0),rankdata(pp,axis=0))[0],
                    'scope':'Full-training composition association, not causal evidence or validation'})
        pd.DataFrame(correlations).to_csv(folder/'cell_composition_associations.csv',index=False)
        # Patient bootstrap repeats affect stability only; they never enter OOF prediction fitting.
        bootstrap=[];rng=np.random.default_rng(20261104)
        training=full[data['primary']] if choice.group=='A' else full
        units=np.unique(data['patients'][training])
        for repeat in range(5):
            draw=rng.choice(units,len(units),replace=True);ix=[];names=[]
            for j,u in enumerate(draw):
                subset=training[data['patients'][training]==u];ix.extend(subset.tolist());names.extend([f'{u}_draw{j}']*len(subset))
            ix=np.asarray(ix);names=np.asarray(names)
            target=TargetSpace(data['y'][ix],data['raw'][ix],data['negative'][ix],names)
            program=ProgramSpace(family,rank,target,data['y'][ix],names)
            a=loading/np.maximum(np.linalg.norm(loading,axis=1,keepdims=True),1e-12)
            b=program.full_loadings();b=b/np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-12)
            similarity=np.abs(a@b.T);row,col=linear_sum_assignment(-similarity)
            ari=None
            if family=='Coexpression':
                label1=np.full(6175,-1);label2=label1.copy()
                label1[model.target.mask]=model.program.labels;label2[target.mask]=program.labels
                common=(label1>=0)&(label2>=0);ari=adjusted_rand_score(label1[common],label2[common])
            for a_id,b_id in zip(row,col):bootstrap.append({'bootstrap':repeat,'program':f'{family}_{a_id+1}',
                'matched_loading_cosine':similarity[a_id,b_id],'coexpression_common_gene_ARI':ari,
                'independent_patients_in_reference':len(units),'note':'Resampled existing patients; not new biological replicates'})
        pd.DataFrame(bootstrap).to_csv(folder/'patient_bootstrap_loading_stability.csv',index=False)
        for k in range(rank):
            for sign in ['positive','negative']:
                selected_genes=set(data['genes'][gene_query(loading[k],sign)])&universe
                if len(selected_genes)<3 or not universe:continue
                entries=[]
                for term,gene_set in terms.items():
                    hits=gene_set&universe
                    if not 5<=len(hits)<=1000:continue
                    overlap=hits&selected_genes
                    entries.append({'program':f'{family}_{k+1}','sign':sign,'term':term,
                        'query_genes':len(selected_genes),'term_genes_measured':len(hits),'overlap':len(overlap),
                        'background_measured_annotated':len(universe),'overlap_genes':';'.join(sorted(overlap)),
                        'pvalue':hypergeom.sf(len(overlap)-1,len(universe),len(hits),len(selected_genes))})
                if entries:
                    df=pd.DataFrame(entries);df['BH_q']=bh(df.pvalue.to_numpy());all_enrich.append(df)
        reference.append({'family':family,'encoder':choice.encoder,'group':choice.group,'rank':rank,
            'diagnostic_primary_MSE_ratio':float(choice.primary_MSE_ratio),
            'ratio_CI_low':float(choice.ratio_CI_low),'ratio_CI_high':float(choice.ratio_CI_high),
            'median_patient_bootstrap_loading_cosine':float(pd.DataFrame(bootstrap).matched_loading_cosine.median()),
            'reference_uses_all_training_data':True,'independent_validation':False})
        print(f'{family} reference, bootstrap and GO enrichment completed',flush=True)
    if all_enrich:pd.concat(all_enrich,ignore_index=True).to_csv(O/'program_GO_enrichment.csv.gz',index=False,compression='gzip')
    else:pd.DataFrame(columns=['program','term','pvalue','BH_q']).to_csv(O/'program_GO_enrichment_UNAVAILABLE.csv',index=False)
    pd.DataFrame(reference).to_csv(O/'program_reference_summary.csv',index=False)
    tr.dump(O/'program_interpretation_limits.json',{'OOF':'Only fold-specific program predictions are validation; no pooling of mismatched component identities',
        'references':'Full-training scores/loadings/tissue and composition associations are descriptive, not held-out accuracy',
        'enrichment_background':len(universe),'measured_RNA_panel':6175,
        'GO_library':str(resource) if resource else None,
        'query_rule':'Within each sign, genes carrying first 50% of squared loading mass; full loadings retained',
        'mechanism':'No LNM mechanism is inferred from program prediction or enrichment'})

def main():
    data=tr.load_data();tr.DATA=data
    run=Path(json.loads((T/'active_run_all_cells_0.95_0.1.json').read_text(encoding='utf-8'))['run']);tr.RUN=run
    summarize_fold_programs(data,run);reference_programs(data,run)
if __name__=='__main__':
    with threadpool_limits(limits=2):main()
