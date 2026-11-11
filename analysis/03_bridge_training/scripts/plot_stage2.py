"""Real-data figures only; each figure exported separately as editable PDF/SVG and PNG."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from PIL import Image
import openslide
import fitz
from scipy.stats import rankdata
import train_bridge as tr

R=tr.R;T=tr.T;G=R/'04_gene_predictability';P=R/'05_molecular_programs';O=R/'06_figures/bridge_figures'
S=O/'source_data';S.mkdir(exist_ok=True)
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','font.size':9,
    'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':10})
COL={'A':'#0072B2','B':'#009E73','C':'#D55E00'}
CAPTIONS=[]

def save(fig,name,caption):
    fig.tight_layout()
    for extension in ['pdf','svg','png']:fig.savefig(O/f'{name}.{extension}',dpi=300,bbox_inches='tight')
    plt.close(fig);CAPTIONS.append({'figure':name,'caption':caption})

def case_means(y,p,patients):
    names=np.unique(patients)
    return names,np.stack([y[patients==u].mean(0) for u in names]),np.stack([p[patients==u].mean(0) for u in names])

def main():
    data=tr.load_data();card=json.loads((R/'07_reports/stage2_primary_results.json').read_text(encoding='utf-8'))
    comparison=pd.read_csv(T/'bridge_model_comparison.csv');genes=pd.read_csv(G/'gene_predictability.csv')
    primary=genes.loc[genes.scope.eq('primary')].copy();broad=genes.loc[genes.scope.eq('broad')]
    selected=np.load(T/'main_nested_OOF.npz');prediction=selected['prediction']
    manifest=pd.read_csv(R/'fov_manifest.csv').set_index('FOV_ID').reindex(data['FOV'])
    qc=pd.read_csv(G/'gene_qc.csv');patients=pd.read_csv(T/'bridge_patient_splits.csv')
    fig,axes=plt.subplots(1,3,figsize=(12,3.7))
    categories=data['m'].tissue_category.value_counts().sort_values()
    axes[0].barh(categories.index,categories.values,color='#0072B2');axes[0].set_xlabel('Eligible measured FOVs')
    axes[0].set_title('375 FOVs / 26 patients')
    axes[1].bar(patients.Patient_ID,patients.FOVs,color='#56B4E9')
    axes[1].bar(patients.Patient_ID,patients.primary_FOVs,color='#D55E00');axes[1].tick_params(axis='x',rotation=90,labelsize=6)
    axes[1].set(ylabel='FOVs per patient',title='154 primary-core FOVs / 24 patients')
    axes[2].hist(qc.cell_detection_fraction,bins=35,color='#009E73');axes[2].set(xlabel='Raw cell-level RNA detection fraction',ylabel='RNA genes',title='All 6,175 RNA targets audited')
    qc.to_csv(S/'FigureA_gene_QC.csv',index=False)
    save(fig,'FigureA_FOV_patient_and_RNA_QC','375 eligible FOVs from 26 confirmed patients. Primary-core evaluation includes 154 FOVs/24 patients. Tissue categories are specimen labels, not pixel-level pathology. Full-panel QC is descriptive; all fitted filters are training-only.')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    ax=axes[0];ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    labels=[('Native H&E FOV\n~512 um',.05,.65),('16 physical patches\n224 px / ~112 um',.52,.65),
            ('Frozen encoder\nResNet50 / Phikon',.05,.25),('FOV pooling -> RNA\npatient-nested CV',.52,.25)]
    for text,x,y in labels:
        ax.add_patch(FancyBboxPatch((x,y),.40,.20,boxstyle='round,pad=0.02',facecolor='#EAF3F8',edgecolor='#0072B2'))
        ax.text(x+.20,y+.10,text,ha='center',va='center',fontsize=9)
    for a,b in [((.45,.75),(.50,.75)),((.72,.63),(.25,.47)),((.45,.35),(.50,.35))]:ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#333333'})
    matched=comparison.loc[comparison.family.eq('Ridge')]
    for j,encoder in enumerate(['ResNet50','Phikon']):
        a=matched.loc[matched.encoder.eq(encoder)].sort_values('group');x=np.arange(3)+(j-.5)*.18
        axes[1].errorbar(x,a.primary_MSE_ratio,yerr=[a.primary_MSE_ratio-a.ratio_CI_low,a.ratio_CI_high-a.primary_MSE_ratio],fmt='o',capsize=3,label=encoder)
    axes[1].axhline(1,color='gray',ls='--');axes[1].set_xticks(range(3),['A: primary','B: broad','C: refined'])
    axes[1].set(ylabel='Primary RNA MSE / training mean',title='Matched Ridge comparison, n=24 patients');axes[1].legend()
    axes[1].text(.02,.02,'UNI / CONCH: not run (gated access)',transform=axes[1].transAxes,fontsize=8)
    matched.to_csv(S/'FigureB_matched_encoder_Ridge.csv',index=False)
    save(fig,'FigureB_architecture_and_encoders','Frozen encoders use identical native physical patch scale. Encoder performance is compared with matched Ridge models and inherited patient folds. Error bars are conditional patient-bootstrap 95% intervals. UNI/CONCH have no fabricated scores.')
    pt=pd.read_csv(T/'primary_per_patient_model_errors.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,encoder in zip(axes,['ResNet50','Phikon']):
        block=pt.loc[pt.model_id.isin([f'{encoder}_HSV_mean_{g}_Ridge' for g in ['A','B','C']])].copy()
        block['group']=block.model_id.str.split('_').str[-2];pivot=block.pivot(index='Patient_ID',columns='group',values='MSE_ratio')
        for _,row in pivot.iterrows():ax.plot([0,1,2],row[['A','B','C']],color='#B0B0B0',lw=.6,alpha=.7)
        for j,g in enumerate(['A','B','C']):ax.scatter(np.repeat(j,len(pivot)),pivot[g],s=13,color=COL[g],zorder=3)
        ax.axhline(1,color='black',ls='--',lw=.8);ax.set_xticks(range(3),['A: primary','B: broad','C: refined'])
        ax.set(title=encoder+' / Ridge',ylabel='Per-patient RNA MSE ratio');ax.set_ylim(bottom=0)
        pivot.to_csv(S/f'FigureC_{encoder}_paired_patient_errors.csv')
    save(fig,'FigureC_FOV_training_strategy_comparison','All three strategies are evaluated on the same 24 held-primary patients. Lines connect the same patient. The 154 primary FOVs are not independent patient replications. Full paired contrasts and multiplicity-adjusted tests are retained.')
    fig,axes=plt.subplots(1,2,figsize=(12,5.5))
    families=list(tr.CONFIG['configs'])
    for ax,encoder in zip(axes,['ResNet50','Phikon']):
        for gi,group in enumerate(['A','B','C']):
            a=comparison.loc[comparison.encoder.eq(encoder)&comparison.group.eq(group)].set_index('family').reindex(families)
            yy=np.arange(len(families))+(gi-1)*.2
            ax.errorbar(a.primary_MSE_ratio,yy,xerr=[a.primary_MSE_ratio-a.ratio_CI_low,a.ratio_CI_high-a.primary_MSE_ratio],fmt='o',ms=3,capsize=2,color=COL[group],label=group)
        ax.axvline(1,color='gray',ls='--');ax.set_yticks(range(len(families)),families);ax.invert_yaxis()
        ax.set(title=encoder,xlabel='Primary RNA MSE ratio (lower is better)');ax.legend(title='Training group',fontsize=7)
    comparison.to_csv(S/'FigureD_all_main_model_comparisons.csv',index=False)
    save(fig,'FigureD_regression_comparison','42 prespecified encoder/strategy/regression combinations. All error bars use the same patient unit. These fixed-family OOF summaries are diagnostic and selection-exposed; the fully nested adaptive selector is separately reported as the primary pipeline.')
    top=primary.sort_values('Spearman_patient_mean',ascending=False).head(2)
    fig,axes=plt.subplots(1,3,figsize=(12,3.7))
    axes[0].hist(primary.Spearman_patient_mean.dropna(),bins=np.linspace(-1,1,41),alpha=.65,label='Primary patients (n=24)',color='#0072B2')
    axes[0].hist(broad.Spearman_patient_mean.dropna(),bins=np.linspace(-1,1,41),alpha=.45,label='Broad patients (n=26)',color='#009E73')
    axes[0].set(xlabel='Patient-mean OOF Spearman rho',ylabel='RNA genes');axes[0].legend(fontsize=7)
    _,yy,pp=case_means(data['y'][data['primary']],prediction[data['primary']],data['patients'][data['primary']])
    for ax,(_,row) in zip(axes[1:],top.iterrows()):
        j=np.flatnonzero(data['genes']==row.gene)[0]
        ax.scatter(yy[:,j],pp[:,j],s=22,color='#0072B2');lo=min(yy[:,j].min(),pp[:,j].min());hi=max(yy[:,j].max(),pp[:,j].max())
        ax.plot([lo,hi],[lo,hi],color='gray',ls='--');ax.set(xlabel='Measured patient-mean log1p(CPM)',ylabel='OOF image-inferred log1p(CPM)',title=f'{row.gene}: rho={row.Spearman_patient_mean:.2f}')
        ax.text(.02,.02,'Exploratory example; not a stable candidate',fontsize=7,transform=ax.transAxes)
    top.to_csv(S/'FigureE_exploratory_example_genes.csv',index=False)
    save(fig,'FigureE_gene_correlations_and_OOF_scatter','Correlations use 24 or 26 independent patient means, not hundreds of FOVs as independent replicates. Scatter examples come from exploratory rankings; no gene passed the full primary stability rule. Constant-expression correlations remain undefined.')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    rank=primary.sort_values('Spearman_patient_mean',ascending=False).head(15)
    gene_index=pd.Index(data['genes']).get_indexer(rank.gene)
    rng=np.random.default_rng(20261104);draw=rng.integers(0,len(yy),size=(2000,len(yy)))
    aa=rankdata(yy[draw][:,:,gene_index],axis=1);bb=rankdata(pp[draw][:,:,gene_index],axis=1)
    aa-=aa.mean(1,keepdims=True);bb-=bb.mean(1,keepdims=True)
    den=np.sqrt((aa*aa).sum(1)*(bb*bb).sum(1))
    boot=np.divide((aa*bb).sum(1),den,out=np.full(den.shape,np.nan),where=den>0)
    rank=rank.copy();rank['rho_bootstrap_CI_low']=np.nanquantile(boot,.025,axis=0);rank['rho_bootstrap_CI_high']=np.nanquantile(boot,.975,axis=0)
    view=rank.iloc[::-1];axes[0].barh(view.gene,view.Spearman_patient_mean,color='#0072B2',alpha=.8)
    axes[0].hlines(np.arange(len(view)),view.rho_bootstrap_CI_low,view.rho_bootstrap_CI_high,color='black',lw=1)
    axes[0].set(xlabel='Patient-mean OOF Spearman rho',title=f'Exploratory ranking; stable genes = {int(primary.stable_exploratory_candidate.sum())}')
    adjusted=pd.read_csv(G/'composition_adjusted_gene_predictability.csv').query("scope=='primary'").set_index('gene').reindex(primary.gene)
    axes[1].scatter(primary.Spearman_patient_mean,adjusted.composition_adjusted_patient_Spearman,s=4,alpha=.25,color='#009E73')
    axes[1].plot([-1,1],[-1,1],color='gray',ls='--');axes[1].set(xlabel='Unadjusted patient OOF rho',ylabel='Crossfit composition-adjusted rho',title='Measured composition/depth sensitivity')
    rank.to_csv(S/'FigureF_exploratory_gene_rank.csv',index=False)
    save(fig,'FigureF_gene_stability_and_composition','Rankings do not imply a reliable gene shortlist. Composition diagnostics independently residualize truth and image predictions using training-patient fits. Composition/depth are measured CosMx covariates, not deployable WSI-only inputs or causal evidence.')
    programs=pd.read_csv(P/'program_model_comparison.csv');refs=pd.read_csv(P/'program_reference_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for gi,g in enumerate(['A','B','C']):
        a=programs.loc[programs.encoder.eq('ResNet50')&programs.group.eq(g)].set_index('family').reindex(['PCA','NMF','Coexpression'])
        xx=np.arange(3)+(gi-1)*.18
        axes[0].errorbar(xx,a.standardized_program_MSE_ratio,yerr=[a.standardized_program_MSE_ratio-a.ratio_CI_low,a.ratio_CI_high-a.standardized_program_MSE_ratio],fmt='o',ms=4,capsize=3,color=COL[g],label=g)
    axes[0].axhline(1,color='gray',ls='--');axes[0].set_xticks(range(3),['PCA','NMF','Coexpression'])
    axes[0].set(ylabel='Training-standardized program score MSE ratio',title='Secondary latent endpoint / 24 patients');axes[0].legend(title='Group',fontsize=7)
    box=[]
    for family in ['PCA','NMF','Coexpression']:
        b=pd.read_csv(P/f'{family}_training_reference/patient_bootstrap_loading_stability.csv');box.append(b.matched_loading_cosine)
    axes[1].boxplot(box,tick_labels=['PCA','NMF','Coexpression'],showfliers=True);axes[1].set_ylim(0,1.05)
    axes[1].set(ylabel='Matched gene-loading cosine',title='Patient-bootstrap loading stability')
    programs.to_csv(S/'FigureG_program_secondary_endpoint.csv',index=False)
    save(fig,'FigureG_program_prediction_and_loading_stability','Program-score prediction is a secondary endpoint distinct from whole-panel gene reconstruction. Basis/rank and score standardization are learned within training folds; component identities are not naively pooled across folds. None of the 18 families passed the program multiplicity/stability criterion. Loading stability is not prediction validity.')
    gene=top.iloc[0].gene;j=np.flatnonzero(data['genes']==gene)[0]
    geo=json.loads((R/'01_registration/cosmx_fovs_author_confirmed.geojson').read_text(encoding='utf-8'))
    svs=openslide.OpenSlide(r'__ESCC_COSMX_HE_WSI__')
    thumbnail=svs.get_thumbnail((1600,1600));width,height=svs.dimensions;svs.close()
    sx=thumbnail.width/width;sy=thumbnail.height/height
    polys={int(f['properties']['FOV_ID']):np.asarray(f['geometry']['coordinates'][0])*[sx,sy] for f in geo['features']}
    truth=data['y'][:,j];inferred=prediction[:,j];norm=Normalize(min(truth.min(),inferred.min()),max(truth.max(),inferred.max()))
    fig,axes=plt.subplots(1,2,figsize=(12,5.5))
    for ax,values,title in zip(axes,[truth,inferred],['Measured CosMx FOV expression','OOF image-inferred FOV expression']):
        ax.imshow(thumbnail);shapes=[Polygon(polys[int(fid)],closed=True) for fid in data['FOV']]
        collection=PatchCollection(shapes,cmap='viridis',norm=norm,alpha=.78,edgecolor='black',linewidth=.12)
        collection.set_array(values);ax.add_collection(collection);ax.axis('off');ax.set_title(title)
    fig.colorbar(collection,ax=axes.ravel().tolist(),shrink=.65,label=gene+' log1p(CPM)')
    fig.suptitle('Exploratory gene example; FOV aggregates only; no stable-gene claim',fontsize=10)
    # Avoid tight_layout reflowing a shared colorbar on this figure.
    for extension in ['pdf','svg','png']:fig.savefig(O/f'FigureH_measured_and_OOF_FOV_expression.{extension}',dpi=300,bbox_inches='tight')
    plt.close(fig);CAPTIONS.append({'figure':'FigureH_measured_and_OOF_FOV_expression',
       'caption':'Same color scale for measured and image-inferred log1p(CPM). Only the actual 375 eligible measured FOV polygons are colored. Unassayed core regions receive no measured labels. Predictions are OOF and FOV-level, not measured cell expression. The illustrative gene did not pass stability; H&E thumbnail is display background only.'})
    pd.DataFrame({'FOV_ID':data['FOV'],'gene':gene,'measured_logCPM':truth,'OOF_image_inferred_logCPM':inferred}).to_csv(S/'FigureH_FOV_expression_values.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    masks=pd.read_csv(T/'image_mask_aggregation_sensitivity.csv')
    for group in ['A','B','C']:
        a=masks.loc[masks.encoder.eq('ResNet50')&masks.group.eq(group)].set_index('aggregation').reindex(['HSV_mean','all16_mean','OD_soft_mean'])
        axes[0].plot(range(3),a.primary_MSE_ratio,'o-',color=COL[group],label=group)
    axes[0].axhline(1,color='gray',ls='--');axes[0].set_xticks(range(3),['HSV','All 16','Relative OD'])
    axes[0].set(ylabel='Primary gene MSE ratio',title='Pooling sensitivity / ResNet50 Ridge');axes[0].legend()
    null=pd.read_csv(T/'patient_permutation_whole_panel_results.csv')
    axes[1].hist(null.loc[null.permutation_id.gt(0),'whole_panel_MSE_ratio'],bins=30,color='#999999')
    axes[1].axvline(null.iloc[0].whole_panel_MSE_ratio,color='#D55E00',label='Unpermuted patient control')
    axes[1].axvline(1,color='black',ls='--',label='Training mean baseline');axes[1].legend(fontsize=7)
    axes[1].set(xlabel='Whole-panel patient-level Ridge MSE ratio',ylabel='Permutations',title='1,000 full patient-profile refits')
    save(fig,'SupplementaryS1_pooling_and_patient_permutation','Per-patch relative OD is corrected to each image own white reference; all16 pooling tests weak-tissue exclusions. The patient-aggregate permutation control uses a fixed auxiliary Ridge, not the full adaptive model-selection pipeline. Beating shuffled labels does not imply beating the training mean.')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    rna=pd.read_csv(T/'RNA_annotation_sensitivity.csv')
    for name,label,color in [(card['model_id'],'Primary Ridge','#0072B2'),('ResNet50_HSV_mean_B_MLP','Broad MLP','#D55E00')]:
        a=rna.loc[rna.model_id.eq(name)].set_index('RNA_stratum').reindex(['all_cells','annotated','unannotated'])
        axes[0].plot(range(3),a.primary_MSE_ratio,'o-',label=label,color=color)
    axes[0].set_xticks(range(3),['All cells','Annotated','Unannotated']);axes[0].axhline(1,color='gray',ls='--');axes[0].legend(fontsize=7)
    axes[0].set(ylabel='Within-stratum primary RNA MSE ratio',title='RNA annotation sensitivity')
    full=pd.read_csv(T/'full_embedding_capacity_sensitivity/v2/full_embedding_Ridge_comparison.csv')
    axes[1].errorbar(full.primary_MSE_ratio,np.arange(len(full)),xerr=[full.primary_MSE_ratio-full.ratio_CI_low,full.ratio_CI_high-full.primary_MSE_ratio],fmt='o',capsize=3,color='#009E73')
    axes[1].set_yticks(range(len(full)),full.model_id.str.replace('_Ridge','').str.replace('_FULL_',' / '),fontsize=8)
    axes[1].axvline(1,color='gray',ls='--');axes[1].set(xlabel='Primary gene MSE ratio',title='Exploratory full-embedding capacity check')
    save(fig,'SupplementaryS2_RNA_strata_and_full_embedding','Annotation status is not assumed to define quality. Stratum-specific RNA and negative probes are exact aggregates. Full-embedding Ridge is an explicitly exploratory capacity sensitivity added after primary results; it does not replace the frozen primary pipeline or establish reliable individual genes.')
    pd.DataFrame(CAPTIONS).to_csv(O/'figure_captions.csv',index=False)
    (O/'图注.md').write_text('# 第二阶段图注\n\n'+'\n\n'.join('## '+x['figure']+'\n\n'+x['caption'] for x in CAPTIONS),encoding='utf-8')
    render=O/'rendered_QA';render.mkdir(exist_ok=True);qa=[];thumbnails=[]
    for path in sorted(O.glob('*.pdf')):
        with fitz.open(path) as doc:
            page=doc[0];assert len(doc)==1 and page.get_text().strip()
            for block in page.get_text('dict')['blocks']:
                if 'lines' not in block:continue
                for line in block['lines']:
                    for span in line['spans']:
                        x0,y0,x1,y1=span['bbox'];assert x0>=-1 and y0>=-1 and x1<=page.rect.width+1 and y1<=page.rect.height+1,(path,span)
            png=render/(path.stem+'.png');page.get_pixmap(matrix=fitz.Matrix(1.2,1.2)).save(png)
            qa.append({'pdf':path.name,'pages':len(doc),'fonts':len(page.get_fonts()),'vector_drawings':len(page.get_drawings()),'text_bounds_ok':True,'rendered':True})
            im=Image.open(png).convert('RGB');im.thumbnail((780,500));thumbnails.append(im)
    for batch in range((len(thumbnails)+3)//4):
        sheet=Image.new('RGB',(1560,1000),'white')
        for i,im in enumerate(thumbnails[batch*4:batch*4+4]):sheet.paste(im,((i%2)*780,(i//2)*500))
        sheet.save(render/f'contact_{batch+1}.jpg',quality=94)
    tr.dump(O/'PDF_structural_QA.json',qa)
    print(f'Exported {len(CAPTIONS)} real-data figures as independent PDF/SVG/PNG and rendered all PDFs',flush=True)

if __name__=='__main__':main()
