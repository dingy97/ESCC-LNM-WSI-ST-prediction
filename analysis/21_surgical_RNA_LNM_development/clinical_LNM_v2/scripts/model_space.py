"""Fixed Stage21 search spaces: every count crossed with every coarse profile."""
import os,sys
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(r'__ESCC_PROJECT_ROOT__');OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'14_RNA_LNM_classifier_optimization/scripts'))
import rna_experiment as r
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
CLASSIFIERS=r.CLASSIFIERS;SELECTORS=r.SELECTORS;COUNTS=[5,10,20,30,50,100]

def build(name,p,seed):
 p=dict(p)
 if name=='Logistic Regression':return LogisticRegression(solver='liblinear',max_iter=10000,random_state=seed,**p)
 if name=='SVM':return SVC(probability=False,random_state=seed,**p)
 if name in ['Random Forest','Extra Trees']:
  cls=RandomForestClassifier if name=='Random Forest' else ExtraTreesClassifier
  return cls(n_jobs=1,random_state=seed,**p)
 if name=='MLP':return MLPClassifier(solver='lbfgs',max_iter=6000,max_fun=60000,random_state=seed,early_stopping=False,**p)
 if name=='LightGBM':
  from lightgbm import LGBMClassifier
  return LGBMClassifier(n_jobs=1,random_state=seed,verbosity=-1,deterministic=True,force_col_wise=True,**p)
 if name=='KNN':return KNeighborsClassifier(n_jobs=1,**p)
 if name=='GBDT':return GradientBoostingClassifier(random_state=seed,**p)
 if name=='Gaussian Naive Bayes':return GaussianNB(**p)
 if name=='CatBoost':
  from catboost import CatBoostClassifier
  return CatBoostClassifier(thread_count=1,random_seed=seed,verbose=False,allow_writing_files=False,**p)
 if name=='AdaBoost':
  depth=p.pop('base_depth',1);leaf=p.pop('base_min_samples_leaf',1)
  return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=leaf,random_state=seed),random_state=seed,**p)
 if name=='XGBoost':
  from xgboost import XGBClassifier
  return XGBClassifier(n_jobs=1,random_state=seed,tree_method='hist',eval_metric='logloss',**p)
 raise ValueError(name)

def profiles(name):
 if name=='Logistic Regression':return [{'C':c,'penalty':'l2'} for c in [.01,.1,1.]]
 if name=='SVM':return [{'C':c,'kernel':'rbf','gamma':'scale'} for c in [.1,1.,10.]]
 if name in ['Random Forest','Extra Trees']:return [{'n_estimators':200,'max_depth':d,'min_samples_leaf':4,'max_features':'sqrt'} for d in [3,6,None]]
 if name=='MLP':return [{'hidden_layer_sizes':h,'alpha':al} for h,al in [([16],1.),([32],.1),([32,16],.01)]]
 if name=='LightGBM':return [{'n_estimators':150,'num_leaves':l,'min_child_samples':10,'learning_rate':.03,'reg_lambda':10.} for l in [3,7,15]]
 if name=='KNN':return [{'n_neighbors':k,'weights':'distance','p':2} for k in [3,7,15]]
 if name=='GBDT':return [{'n_estimators':150,'max_depth':d,'min_samples_leaf':5,'learning_rate':.03} for d in [1,2,3]]
 if name=='Gaussian Naive Bayes':return [{'var_smoothing':v} for v in [1e-8,.001,.1]]
 if name=='CatBoost':return [{'iterations':200,'depth':d,'learning_rate':.03,'l2_leaf_reg':10.} for d in [3,4,5]]
 if name=='AdaBoost':return [{'n_estimators':150,'learning_rate':l,'base_depth':1,'base_min_samples_leaf':1} for l in [.05,.1,.5]]
 if name=='XGBoost':return [{'n_estimators':150,'max_depth':d,'learning_rate':.03,'reg_lambda':10.,'subsample':1.,'colsample_bytree':1.} for d in [1,2,3]]
 raise ValueError(name)

def sample(name,rng):
 pick=lambda x:x[int(rng.integers(len(x)))];log=lambda lo,hi:float(10**rng.uniform(lo,hi))
 if name=='Logistic Regression':return {'C':log(-3,2),'penalty':pick(['l1','l2']),'class_weight':pick([None,'balanced'])}
 if name=='SVM':
  p={'C':log(-2,2),'kernel':pick(['rbf','linear']),'class_weight':pick([None,'balanced'])}
  if p['kernel']=='rbf':p['gamma']=log(-4,0)
  return p
 if name in ['Random Forest','Extra Trees']:return {'n_estimators':pick([150,300,500]),'max_depth':pick([3,5,8,None]),'min_samples_leaf':pick([2,4,8]),'max_features':pick(['sqrt',.5,1.]),'class_weight':pick([None,'balanced'])}
 if name=='MLP':return {'hidden_layer_sizes':pick([[8],[16],[32],[64],[32,16]]),'alpha':log(-3,1)}
 if name=='LightGBM':return {'n_estimators':pick([100,200,400]),'num_leaves':pick([3,7,15]),'min_child_samples':pick([5,10,20]),'learning_rate':log(-2,-.7),'reg_lambda':log(-1,1.5)}
 if name=='KNN':return {'n_neighbors':pick([3,5,7,9,11,15,21]),'weights':pick(['uniform','distance']),'p':pick([1,2])}
 if name=='GBDT':return {'n_estimators':pick([100,200,300]),'max_depth':pick([1,2,3]),'min_samples_leaf':pick([3,5,10]),'learning_rate':log(-2,-.7),'subsample':float(rng.uniform(.7,1))}
 if name=='Gaussian Naive Bayes':return {'var_smoothing':log(-10,-1)}
 if name=='CatBoost':return {'iterations':pick([100,250,400]),'depth':pick([2,3,4,5,6]),'learning_rate':log(-2,-.7),'l2_leaf_reg':log(-1,1.5)}
 if name=='AdaBoost':return {'n_estimators':pick([50,100,200,400]),'learning_rate':log(-2,0),'base_depth':pick([1,2]),'base_min_samples_leaf':pick([1,3,5])}
 if name=='XGBoost':return {'n_estimators':pick([100,200,400]),'max_depth':pick([1,2,3,5]),'min_child_weight':pick([1,5,10]),'learning_rate':log(-2,-.7),'reg_lambda':log(-1,1.5),'reg_alpha':pick([0.,1.]),'subsample':float(rng.uniform(.7,1)),'colsample_bytree':float(rng.uniform(.7,1))}
 raise ValueError(name)

def coarse(name):return [{'selector':s,'count':k,'config':p} for s in SELECTORS for k in COUNTS for p in profiles(name)]
