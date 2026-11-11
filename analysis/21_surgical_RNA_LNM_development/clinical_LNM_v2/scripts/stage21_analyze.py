"""All twelve models: training-only ranking, held-out performance and editable figures."""
import os,sys
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
import json,hashlib,itertools,textwrap
from pathlib import Path
import train_all_12 as t
from model_space import ROOT,OUT,CLASSIFIERS,SELECTORS
import numpy as np,pandas as pd,joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve,roc_auc_score,average_precision_score,brier_score_loss
from sklearn.calibration import calibration_curve

def save(name,rows):
 d=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows);d.to_csv(OUT/'03_internal_evaluation'/f'{name}.csv',index=False,encoding='utf-8-sig',float_format='%.17g');return d
def export(fig,name):
 for ext in ['pdf','svg','png']:fig.savefig(OUT/f'06_figures/{name}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
 plt.close(fig)
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def interval(y,p,h):
 result={**t.metrics(y,p,h),**t.r.metrics(y,h)}
 rng=np.random.default_rng(20261107);pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0);boot=[]
 for _ in range(2000):
  ix=np.r_[rng.choice(pos,len(pos),True),rng.choice(neg,len(neg),True)]
  boot.append([roc_auc_score(y[ix],p[ix]),average_precision_score(y[ix],p[ix]),brier_score_loss(y[ix],p[ix])])
 ci=np.quantile(boot,[.025,.975],axis=0)
 for i,k in enumerate(['AUROC','AUPRC','Brier']):result[k+'_CI_low']=float(ci[0,i]);result[k+'_CI_high']=float(ci[1,i])
 return result

def run():
 hold=OUT/'logs/label_semantics_hold.json'
 if hold.exists() and json.loads(hold.read_text(encoding='utf-8')).get('status')!='RESOLVED':
  raise RuntimeError('Outcome semantics unresolved; numerical folder-target results cannot be promoted to formal LNM results')
 complete=OUT/'02_training/training_complete.json'
 assert complete.exists(),'Do not turn partial model searches into a final comparison'
 lock=json.loads((OUT/'config/final_model_lock.json').read_text(encoding='utf-8'));winner=lock['winner_classifier']
 assert t.sha(Path(lock['winner_directory'])/'model.joblib')==lock['model_SHA256']
 units=json.loads((OUT/'02_training/all_units_complete.json').read_text(encoding='utf-8'))
 from fixed_candidate_sensitivity import run as run_sensitivity
 run_sensitivity()
 predictions=pd.read_parquet(OUT/'03_internal_evaluation/all_12_patient_OOF.parquet')
 primary=predictions[predictions.target==.5].copy()
 assert primary.groupby('classifier').size().eq(160).all() and not primary.duplicated(['classifier','Patient_ID']).any()
 scores=[];folds=[];rocrows=[];prrows=[];calrows=[]
 for name in CLASSIFIERS:
  d=primary[primary.classifier==name].sort_values('Patient_ID');y=d.LNM.to_numpy(int);p=d.probability.to_numpy();h=d.high_risk.to_numpy(bool)
  scores.append({'classifier':name,**interval(y,p,h)})
  for f,g in d.groupby('outer_fold'):folds.append({'classifier':name,'outer_fold':f,**t.metrics(g.LNM.to_numpy(int),g.probability.to_numpy(),g.high_risk.to_numpy(bool))})
  xx,yy,_=roc_curve(y,p);rocrows.extend({'classifier':name,'FPR':x,'TPR':z} for x,z in zip(xx,yy))
  precision,recall,_=precision_recall_curve(y,p);prrows.extend({'classifier':name,'Recall':x,'Precision':z} for x,z in zip(recall,precision))
  observed,predicted=calibration_curve(y,p,n_bins=8,strategy='quantile');calrows.extend({'classifier':name,'mean_predicted':x,'observed_LNM':z} for x,z in zip(predicted,observed))
 score=save('all_12_outer_OOF_metrics',scores);fold=save('all_12_outer_fold_metrics',folds);roc=save('all_12_ROC_coordinates',rocrows);pr=save('all_12_PR_coordinates',prrows);save('all_12_calibration_curves',calrows)
 comparisons=[]
 for ranked in lock['ranking']:
  name=ranked['classifier'];c=ranked['chosen'];oof=score[score.classifier==name].iloc[0]
  fitted=joblib.load(Path(ranked['directory'])/'model.joblib')
  row={'classifier':name,'best_feature_selector':c['recipe']['selector'],'best_RNA_count':c['recipe']['count'],
   'final_RNA_features':';'.join(c['selected_genes']),'best_hyperparameters':json.dumps(c['recipe']['config'],ensure_ascii=False,sort_keys=True),
   'actual_estimator_parameters':json.dumps(fitted.estimator.get_params(deep=True),ensure_ascii=False,sort_keys=True,default=str),
   'inner_CV_AUROC_mean':c['inner_mean_AUROC'],'inner_CV_AUROC_SD':c['inner_SD_AUROC'],'inner_CV_AUPRC_mean':c['inner_mean_AUPRC'],
   'ranking':ranked['rank'],'attempted_searches':c['attempted_searches'],'unique_valid_trials':c['unique_valid_trials'],
   'invalid_numerical_attempts':c['invalid_numerical_attempts'],'completed_base_search_fits':c['search_classifier_fit_calls'],'selected_final':name==winner}
  for metric in ['AUROC','AUPRC','Accuracy','Sensitivity','Specificity','Precision','F1','Brier']:
   row['inner_pooled_'+metric]=c[metric];row['outer_OOF_'+metric]=float(oof[metric])
  comparisons.append(row)
  if name==winner:
   t.dump(OUT/'07_reports/final_model_parameters_and_genes.json',{'classifier':name,'selector':c['recipe']['selector'],'RNA_count':c['recipe']['count'],
    'genes':c['selected_genes'],'search_parameters':c['recipe']['config'],'actual_estimator_parameters':fitted.estimator.get_params(deep=True),
    'calibration':c['calibration_parameters'],'thresholds':c['thresholds'],'model_SHA256':c['model_SHA256'],'locked_at':lock['locked_at']})
   pd.DataFrame({'gene':c['selected_genes'],'order_in_model':np.arange(1,len(c['selected_genes'])+1)}).to_csv(OUT/'07_reports/final_model_RNA_features.csv',index=False,encoding='utf-8-sig')
  del fitted
 comparison=save('final_12_model_comparison',comparisons)
 comparison.to_csv(OUT/'07_reports/final_12_model_comparison.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
 attempts=[];selections=[];clinical=[];stability=[];counts=[];innerrows=[]
 gene_order=pd.read_csv(ROOT/'14_RNA_LNM_classifier_optimization/inventory/RNA_feature_audit.csv').gene.to_numpy(str)
 for z in units:
  directory=Path(z['directory']);name=z['classifier'];c=z['context'];chosen=z['chosen']
  for a in json.loads((directory/'all_attempts.json').read_text(encoding='utf-8')):
   rec={'cohort':c['cohort'],'outer_fold':c['fold'],'classifier':name,'attempt':a['attempt'],'stage':a['stage'],'status':a['status'],'selector':a['recipe']['selector'],'RNA_count':a['recipe']['count'],'parameters':json.dumps(a['recipe']['config'],sort_keys=True),'candidate_id':a['candidate_id']}
   for k in ['inner_mean_AUROC','inner_SD_AUROC','inner_mean_AUPRC','AUROC','AUPRC','Brier','Accuracy','Sensitivity','Specificity','Precision','F1','classifier_base_fit_calls']:rec[k]=a.get(k,np.nan)
   attempts.append(rec)
  counts.append({'classifier':name,'cohort':c['cohort'],'outer_fold':c['fold'],'count':chosen['recipe']['count'],'selector':chosen['recipe']['selector'],'genes':';'.join(chosen['selected_genes'])})
  if c['cohort']=='main':innerrows.append({'classifier':name,'outer_fold':c['fold'],'inner_mean_AUROC':chosen['inner_mean_AUROC'],'inner_SD_AUROC':chosen['inner_SD_AUROC']})
 for f in range(5):
  candidates=[z for z in units if z['context']['cohort']=='main' and z['context']['fold']==f];best=min(candidates,key=lambda z:t.order(z['chosen']))
  selections.append({'outer_fold':f,'training_inner_selected_classifier':best['classifier'],'RNA_count':best['chosen']['recipe']['count'],'selector':best['chosen']['recipe']['selector']})
 for name in CLASSIFIERS:
  sets=[set(z['chosen']['selected_genes']) for z in units if z['classifier']==name and z['context']['cohort']=='main']
  full=next(z for z in units if z['classifier']==name and z['context']['cohort']=='full_development')
  for gene in gene_order:
   n=sum(gene in g for g in sets);stability.append({'classifier':name,'gene':gene,'selected_outer_folds':n,'frequency':n/5,'selected_full_development':gene in full['chosen']['selected_genes']})
 for (name,target),g in predictions[predictions.target!=.5].groupby(['classifier','target']):
  clinical.append({'classifier':name,'training_target':target,**t.metrics(g.LNM.to_numpy(int),g.probability.to_numpy(),g.high_risk.to_numpy(bool)),**t.r.metrics(g.LNM.to_numpy(int),g.high_risk.to_numpy(bool))})
 attempts=save('all_hyperparameter_search_results',attempts);counts=save('all_12_selected_counts_by_fold',counts);stability=save('all_12_feature_stability',stability)
 frequency=pd.DataFrame(selections).training_inner_selected_classifier.value_counts().reindex(CLASSIFIERS,fill_value=0)
 save('classifier_selection_frequency_counts',[{'classifier':n,'selected_outer_training_folds':int(frequency[n]),'total_folds':5} for n in CLASSIFIERS])
 save('classifier_selection_frequency',selections);save('all_12_training_inner_winners',innerrows);save('all_12_high_sensitivity_workpoints',clinical)
 process=pd.read_parquet(OUT/'03_internal_evaluation/complete_selection_process_OOF.parquet');process_results=[]
 for target,g in process.groupby('target'):process_results.append({'target':target,**interval(g.LNM.to_numpy(int),g.probability.to_numpy(),g.high_risk.to_numpy(bool))})
 save('complete_model_selection_process_OOF_metrics',process_results)
 sensitivity=pd.read_parquet(OUT/'03_internal_evaluation/fixed_candidate_sensitivity_OOF.parquet');sensitivity_scores=[]
 for (cohort,target),group in sensitivity.groupby(['cohort','target']):
  sensitivity_scores.append({'cohort':cohort,'target':target,**interval(group.LNM.to_numpy(int),group.probability.to_numpy(),group.high_risk.to_numpy(bool))})
 save('fixed_candidate_sensitivity_metrics',sensitivity_scores)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
 palette=plt.get_cmap('tab20');colors={name:palette(i) for i,name in enumerate(CLASSIFIERS)}
 fig,ax=plt.subplots(figsize=(12,5));values=[fold[fold.classifier==n].AUROC.to_numpy() for n in CLASSIFIERS]
 ax.boxplot(values,positions=range(12),widths=.55,showfliers=False)
 for i,(name,v) in enumerate(zip(CLASSIFIERS,values)):ax.scatter(i+np.linspace(-.14,.14,5),v,color=colors[name],s=24,zorder=3)
 ax.set(xticks=range(12),xticklabels=CLASSIFIERS,ylim=(0,1),ylabel='Held-out outer-fold AUROC',title='Figure 4A | All 12 classifiers: fixed patient-level outer folds (n=5)');ax.tick_params(axis='x',rotation=40)
 export(fig,'Figure4A_all12_AUROC_distribution')
 abbreviations=['LR','SVM','RF','MLP','LGBM','KNN','GBDT','GNB','ET','CatBoost','AdaBoost','XGB']
 metric_names=['AUROC','AUPRC','Accuracy','Sensitivity','Specificity','Precision','F1','Brier']
 fig,axes=plt.subplots(2,4,figsize=(14,10),constrained_layout=True)
 for ax,k in zip(axes.flat,metric_names):
  values=fold.pivot(index='classifier',columns='outer_fold',values=k).reindex(index=CLASSIFIERS,columns=range(5)).to_numpy()
  color=1-values if k=='Brier' else values
  ax.pcolormesh(color,cmap='viridis',vmin=0,vmax=1,edgecolors='white');ax.invert_yaxis()
  ax.set(xticks=np.arange(5)+.5,xticklabels=np.arange(1,6),yticks=np.arange(12)+.5,yticklabels=abbreviations,title=k+(' (color reversed)' if k=='Brier' else ''),xlabel='Outer test fold')
  for i in range(12):
   for j in range(5):ax.text(j+.5,i+.5,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=6,color='white' if color[i,j]<.55 else 'black')
 fig.suptitle('All twelve classifiers: every held-out fold and metric; no outer-score model selection')
 export(fig,'Supplement_all12_fold_multimetric_performance')
 def curves_plot(data,x,y,title,filename,facets=False):
  fig,axes=plt.subplots(3,4,figsize=(13,9),sharex=True,sharey=True) if facets else plt.subplots(figsize=(9,7))
  for i,name in enumerate(CLASSIFIERS):
   ax=axes.flat[i] if facets else axes;d=data[data.classifier==name];m=score[score.classifier==name].iloc[0]
   ax.plot(d[x],d[y],color=colors[name],linestyle=['-','--',':'][i//4],label=f'{name} ({m.AUROC if x=="FPR" else m.AUPRC:.3f})')
   if facets:ax.set_title(name,fontsize=10)
   ax.set(xlim=(0,1),ylim=(0,1.02),xlabel=x,ylabel=y)
   if x=='FPR':ax.plot([0,1],[0,1],color='gray',linestyle=':',linewidth=.7)
   if facets:ax.legend(frameon=False,fontsize=6,loc='lower right')
  if not facets:axes.legend(frameon=False,fontsize=8,bbox_to_anchor=(1.02,1),loc='upper left')
  fig.suptitle(title);fig.tight_layout();export(fig,filename)
 curves_plot(roc,'FPR','TPR','Figure 4B | Paired patient OOF ROC curves','Figure4B_all12_ROC_overlay')
 curves_plot(roc,'FPR','TPR','All 12 classifiers: held-out ROC facets','Figure4B_all12_ROC_facets',True)
 curves_plot(pr,'Recall','Precision','All 12 classifiers: held-out precision–recall','Supplement_all12_PR_overlay')
 curves_plot(pr,'Recall','Precision','All 12 classifiers: precision–recall facets','Supplement_all12_PR_facets',True)
 columns=['AUROC','AUPRC','Accuracy','Sensitivity','Specificity','Precision','F1','Brier']
 matrix=score.set_index('classifier').loc[CLASSIFIERS,columns].to_numpy();shade=matrix.copy();shade[:,-1]=1-shade[:,-1]
 fig,ax=plt.subplots(figsize=(10,7));mesh=ax.pcolormesh(shade,cmap='viridis',vmin=0,vmax=1,edgecolors='white');ax.invert_yaxis()
 ax.set(xticks=np.arange(8)+.5,xticklabels=columns,yticks=np.arange(12)+.5,yticklabels=CLASSIFIERS,title='Figure 4C | Outer OOF metrics; threshold 0.5; Brier color reversed')
 for i in range(12):
  for j in range(8):ax.text(j+.5,i+.5,f'{matrix[i,j]:.3f}',ha='center',va='center',color='white' if shade[i,j]<.55 else 'black',fontsize=9)
 bar=fig.colorbar(mesh,ax=ax,label='Higher color score is better (Brier uses 1−Brier)');bar.solids.set_rasterized(False);export(fig,'Figure4C_all12_multimetric_heatmap')
 rank=comparison.sort_values('ranking',ascending=False);fig,ax=plt.subplots(figsize=(9,6))
 ax.barh(rank.classifier,rank.inner_CV_AUROC_mean,xerr=rank.inner_CV_AUROC_SD,color=[colors[n] for n in rank.classifier],capsize=3)
 ax.set(xlim=(0,1),xlabel='Full-development training-inner mean AUROC ± fold SD (n=3)',title='Figure 4D | Actual training-only selection ranking; not outer ranking')
 export(fig,'Figure4D_training_inner_selection_ranking')
 chosen=lock['winner'];curve=json.loads((Path(lock['winner_directory'])/'matched_count_curve.json').read_text(encoding='utf-8'))
 curve_df=pd.DataFrame([{'count':z['recipe']['count'],'inner_mean_AUROC':z.get('inner_mean_AUROC',np.nan),'inner_SD_AUROC':z.get('inner_SD_AUROC',np.nan),'inner_mean_AUPRC':z.get('inner_mean_AUPRC',np.nan)} for z in curve]);save('final_model_matched_count_curve',curve_df)
 fig,ax=plt.subplots(figsize=(8,5));ax.errorbar(curve_df['count'],curve_df.inner_mean_AUROC,yerr=curve_df.inner_SD_AUROC,fmt='o-',capsize=3,label='Mean AUROC ± SD (3 inner folds)');ax.plot(curve_df['count'],curve_df.inner_mean_AUPRC,'s--',label='Mean AUPRC')
 ax.axvline(chosen['recipe']['count'],color='#D55E00',linestyle=':',label='Selected count');ax.set(xlabel='RNA count; same selector and hyperparameters',ylabel='Training-inner validation score',ylim=(0,1),title=f'Figure 4E | {winner}: matched count comparison');ax.legend(frameon=False);export(fig,'Figure4E_final_model_feature_count_curve')
 search=attempts[(attempts.cohort=='full_development')&(attempts.classifier==winner)].sort_values('attempt')
 fig,axes=plt.subplots(1,2,figsize=(12,5));axes[0].plot(search.attempt,search.inner_mean_AUROC,'o',markersize=3,alpha=.55,label='Trial inner AUROC');axes[0].plot(search.attempt,search.inner_mean_AUROC.cummax(),color='#D55E00',label='Best so far');axes[0].set(xlabel='Search attempt (all 192 retained)',ylabel='Training-inner mean AUROC',ylim=(0,1),title='A  Hyperparameter optimization trajectory');axes[0].legend(frameon=False,fontsize=8)
 text=f'Classifier: {winner}\nSelector: {chosen["recipe"]["selector"]}\nRNA count: {chosen["recipe"]["count"]}\n\nParameters:\n'+json.dumps(chosen['recipe']['config'],indent=2)+'\n\nImage-inferred RNA features:\n'+textwrap.fill(', '.join(chosen['selected_genes'][:12]),width=58)+('\n(full list in CSV)' if len(chosen['selected_genes'])>12 else '')
 axes[1].axis('off');axes[1].text(0,1,text,transform=axes[1].transAxes,va='top',fontsize=9,wrap=True);fig.suptitle('Figure 4F | Training-only final model and optimization evidence');fig.tight_layout();export(fig,'Figure4F_final_parameters_and_search')
 a=attempts[(attempts.cohort=='full_development')&(attempts.stage=='A_complete_factorial')&(attempts.status=='valid')]
 grid=a.groupby(['classifier','selector']).inner_mean_AUROC.max().unstack().reindex(index=CLASSIFIERS,columns=SELECTORS);save('balanced_stageA_selector_comparison',grid.reset_index())
 fig,ax=plt.subplots(figsize=(12,7));mesh=ax.pcolormesh(grid.to_numpy(),cmap='viridis',vmin=0,vmax=1,edgecolors='white');ax.invert_yaxis();ax.set(xticks=np.arange(9)+.5,xticklabels=SELECTORS,yticks=np.arange(12)+.5,yticklabels=CLASSIFIERS,title='Nine selectors: best training-inner AUROC from equal Stage-A budgets');ax.tick_params(axis='x',rotation=40);bar=fig.colorbar(mesh,ax=ax);bar.solids.set_rasterized(False);export(fig,'Supplement_nine_selector_comparison')
 fig,ax=plt.subplots(figsize=(12,5));c=counts[counts.cohort=='main']
 for i,name in enumerate(CLASSIFIERS):
  v=c[c.classifier==name]['count'].to_numpy();ax.scatter(i+np.linspace(-.15,.15,len(v)),v,color=colors[name]);full=counts[(counts.classifier==name)&(counts.cohort=='full_development')]['count'].iloc[0];ax.scatter(i,full,marker='*',s=100,color='black')
 ax.set(xticks=range(12),xticklabels=CLASSIFIERS,ylabel='Selected RNA count',ylim=(0,110),title='All classifiers: five training-fold selections; star = full development');ax.tick_params(axis='x',rotation=40);export(fig,'Supplement_all12_selected_count_distribution')
 stable=stability[stability.classifier==winner].sort_values(['frequency','gene'],ascending=[False,True]).head(20).sort_values('frequency')
 fig,ax=plt.subplots(figsize=(8,6));ax.barh(stable.gene,stable.frequency,color='#0072B2');ax.set(xlim=(0,1.05),xlabel='Fraction of 5 outer-training fits',title=f'{winner}: image-inferred RNA selection stability');export(fig,'Supplement_final_model_feature_stability')
 fig,axes=plt.subplots(3,4,figsize=(14,12),constrained_layout=True)
 for ax,name in zip(axes.flat,CLASSIFIERS):
  d=stability[stability.classifier==name].sort_values(['frequency','gene'],ascending=[False,True]).head(10).sort_values('frequency')
  ax.barh(d.gene,d.frequency,color=colors[name]);ax.set(xlim=(0,1.05),title=name,xlabel='Selection frequency (5 training folds)');ax.tick_params(axis='y',labelsize=7)
 fig.suptitle('All twelve classifiers: image-inferred RNA selection stability, not molecular validation')
 export(fig,'Supplement_all12_feature_selection_stability')
 fig,axes=plt.subplots(3,4,figsize=(14,10),constrained_layout=True)
 for ax,name in zip(axes.flat,CLASSIFIERS):
  d=attempts[(attempts.cohort=='full_development')&(attempts.classifier==name)].sort_values('attempt')
  ax.scatter(d.attempt,d.inner_mean_AUROC,s=5,alpha=.5,color=colors[name]);ax.plot(d.attempt,d.inner_mean_AUROC.cummax(),color='black',linewidth=1)
  ax.set(ylim=(0,1),xlim=(0,193),title=name,xlabel='Search attempt',ylabel='Training-inner AUROC')
 fig.suptitle('All twelve classifiers: identical planned attempt budgets; duplicate fits explicitly reused')
 export(fig,'Supplement_all12_hyperparameter_search_trajectories')
 # Rationale distinguishes final INNER nomination from outer-family descriptive performance.
 best_auc=float(comparison.inner_CV_AUROC_mean.max());near=comparison[comparison.inner_CV_AUROC_mean>=best_auc-.01].classifier.tolist()
 best_outer=score.sort_values('AUROC',ascending=False).iloc[0];win_oof=score[score.classifier==winner].iloc[0]
 process_primary=next(z for z in process_results if z['target']==.5)
 dataaudit=json.loads((OUT/'00_inventory/audit_summary.json').read_text(encoding='utf-8'))
 resulttable=table(['分类器','外层OOF AUROC','AUPRC','敏感度(.5)','特异度(.5)','Brier','训练内排名'],[[n,f'{score[score.classifier==n].AUROC.iloc[0]:.4f}',f'{score[score.classifier==n].AUPRC.iloc[0]:.4f}',f'{score[score.classifier==n].Sensitivity.iloc[0]:.3f}',f'{score[score.classifier==n].Specificity.iloc[0]:.3f}',f'{score[score.classifier==n].Brier.iloc[0]:.4f}',int(comparison[comparison.classifier==n].ranking.iloc[0])] for n in CLASSIFIERS])
 report=f'''# 第21阶段最终模型选择依据\n\n生成时间：{t.now()}。完整12算法全部完成；所有排名取训练内数据，外层、ESD及TCGA未用于选择。原历史模型不覆盖。\n\n## 逐项回答\n\n1. 最终提名 **{winner}**，由全开发集内层平均AUROC优先的冻结规则选出，值为{chosen['inner_mean_AUROC']:.4f}，SD={chosen['inner_SD_AUROC']:.4f}。次级顺序为平均AUPRC、Brier、AUROC SD、RNA数量、固定哈希平局；无算法名称优先级。\n2. 优势是本次训练内排序中领先；不能把搜索内最高点解释成对其他11种算法已经证明具有统计学优势。\n3. 它的训练内平均AUROC确为最高。外层OOF最高算法为{best_outer['classifier']}（{best_outer['AUROC']:.4f}）；此处仅描述，未用于改提名。\n4. 最终RNA数{chosen['recipe']['count']}，筛选器{chosen['recipe']['selector']}，来自完整交叉搜索及各算法相同的后续细化；最终固定参数下全部六数量已计算。不存在数量与某一学习率绑定。\n5. 参数：`{json.dumps(chosen['recipe']['config'],ensure_ascii=False)}`。每算法每情景192次搜索尝试；实际唯一/有效/数值失败次数均在CSV，重复配方没有虚报为新拟合。\n6. 训练内AUROC距最高≤0.01的竞争模型：{', '.join(near)}。该范围只是描述性接近标记，不是事后重新排序依据。\n7. 五个外层训练折的分类器选择见classifier_selection_frequency.csv；整个12模型选择流程的外层OOF AUROC={process_primary['AUROC']:.4f}，95%条件性CI {process_primary['AUROC_CI_low']:.4f}–{process_primary['AUROC_CI_high']:.4f}。最终算法家族自身OOF AUROC={win_oof['AUROC']:.4f}，这与整个选择流程估计不同，不能混写。\n8. 输入仅为冻结Phikon→Ridge得到的6175维图像推断RNA，再在训练内选择；部署仅需H&E WSI，不输入临床病理因素或实测RNA。图像推断RNA不是已验证的实测表达重建。\n\n## 全部12种算法的真实外层结果\n\n{resulttable}\n\n所有方法使用相同患者划分，5个外层训练部分均重新完整优化全部12分类器。主表含训练内平均和合并指标以及单列外层指标，明确区分。两阶段细化不是只精调前三算法。9筛选器比较图只取均衡的StageA数据。最终特征数量曲线固定筛选器和超参数，评价全部六数量。\n\n## 队列与外部评价限制\n\n外科目录重新核实为{dataaudit['cohorts']['surgical']['mapped_accession_groups']}个病例、{dataaudit['cohorts']['surgical']['WSI_files']}张原图，650张可用。全部患者参与过历史探索，当前外层评价不能描述为全新未暴露盲测。患者结局以作者确认的临床表Lymph node metastasis (+)列为准（100阳性、60阴性）；2例旧文件夹覆盖已在本新分支纠正，历史模型保留；22例桥接监督重叠仍在主分析，这次分类器留出不等于未参与桥接监督。\n\n内镜816张归为61个病理号组。按作者确认，同一病理号任一切片出现在阳性文件夹即为LNM阳性。以冻结模型开展同中心内镜验证，主分析排除已知外科开发和桥接监督重叠。TCGA用于补充跨域评价。两队列的分组、图像质控和结果见冻结模型外部评价报告。\n\n## 冻结文件\n\n`{lock['winner_directory']}/model.joblib`\n\nSHA256：`{lock['model_SHA256']}`。锁定时间：{lock['locked_at']}。概率阈值0.5及训练内95%、98%工作点均保存在chosen/lock中；不得在外部重调。\n\n全部图有可编辑PDF、SVG、PNG；原始统计、全部超参数试验、逐折患者预测、特征集合及拟合事件日志保留。本次属于历史结果已知后的探索性再开发。\n'''
 (OUT/'07_reports/final_model_selection_rationale.md').write_text(report,encoding='utf-8')
 (OUT/'07_reports/internal_12_classifier_report.md').write_text(report,encoding='utf-8')
 methods=f'''# Stage21 manuscript methods and interim results\n\n## Methods\n\nThis historical-results-known exploratory redevelopment used the surgically designated development directory after a read-only accession and label audit. Image-inferred RNA features were generated by the frozen physical-FOV Phikon–Ridge pipeline, without clinical covariates as inputs. All twelve classifiers were optimized in every outer training fold using identical patient-level partitions. The nine ranking methods were fitted only within their corresponding training portions. For each classifier and context, 162 complete factorial configurations crossed nine selectors, six counts and three classifier-specific parameter profiles; 24 fine-search attempts were allocated to every classifier, followed by six matched count evaluations at that classifier's selected selector/parameters. Duplicated recipes were reused and effective fit counts were retained. Three-fold inner validation selected by mean AUROC, followed by mean average precision, Brier score, AUROC SD, feature count and deterministic hash ties. Fixed sigmoid calibration used training-only cross-fitted margins, with selection/scaling refitted in the calibration training portions. No ESD or TCGA results entered optimization.\n\n## Internal results\n\nThe final full-development inner search nominated {winner}, using {chosen['recipe']['selector']} and {chosen['recipe']['count']} image-inferred RNA features. The complete nested model-selection procedure yielded an OOF AUROC of {process_primary['AUROC']:.3f} (conditional bootstrap 95% CI {process_primary['AUROC_CI_low']:.3f}–{process_primary['AUROC_CI_high']:.3f}). Its algorithm-family-specific OOF AUROC was {win_oof['AUROC']:.3f}; this descriptive estimate was not used for nomination. All twelve results and search budgets are retained in the comparison table.\n\n## Same-center ESD and public cross-domain results\n\nThe surgical patient endpoint came from the author-confirmed clinical workbook (100 LNM-positive and 60 LNM-negative patients); two historical folder overrides were corrected in this isolated rerun. The author confirmed that any slide in the positive ESD folder marks the corresponding pathology accession as LNM-positive. Outcomes were assigned before image QC and all eligible H&E slides were pooled within each accession. The primary same-center validation excludes known overlaps with surgical development and CosMx bridge supervision. TCGA is evaluated only after model freezing as a previously exposed public cross-domain cohort. No new blind test is claimed.\n'''
 (OUT/'07_reports/Stage21_methods_results_draft.md').write_text(methods,encoding='utf-8')
 t.dump(OUT/'03_internal_evaluation/analysis_complete.json',{'time':t.now(),'classifiers':12,'OOF_patients_per_classifier':160,'figure_families':11,'winner':winner,'outer_metrics_used_for_selection':False})
 print('ALL_12_ANALYSIS_AND_FIGURES_COMPLETE',winner,flush=True)

if __name__=='__main__':run()
