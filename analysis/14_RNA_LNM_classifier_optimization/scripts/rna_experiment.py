"""RNA-only, nine selectors, twelve classifiers, strictly nested patient CV."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
from datetime import datetime
import sys,json,hashlib,time,warnings,gzip,traceback
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'14_RNA_LNM_classifier_optimization';OLD=ROOT/'10_lnm_sampling_corrected'
sys.path.insert(0,str(OLD/'scripts'));sys.path.insert(0,str(ROOT/'12_stage5_high_sensitivity/scripts'))
import common
from ml_components import CLASSIFIERS,SELECTORS,classifier,near_constant_columns
from clinical_utils import choose_cutoff,metrics
import numpy as np,pandas as pd,joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from scipy.special import logit
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
SEED=20261106;COUNTS=[5,10,20,30,50,100];TARGETS=[.95,.98];LIVE={};TIMINGS=[]
def now():return datetime.now().astimezone().isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def seedof(x):return int.from_bytes(hashlib.sha256(str(x).encode()).digest()[:4],'little')
def dump(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def publish(**kw):
    LIVE.update(kw);LIVE['time']=now();LIVE['patients']=160;LIVE['WSI']=650;LIVE['clinical_safety']='NOT_ADJUDICATED'
    if len(TIMINGS)>=12 and len({x['classifier'] for x in TIMINGS})==12:
        left=0
        for name in CLASSIFIERS:
            a=[x['seconds'] for x in TIMINGS if x['classifier']==name];left+=np.mean(a)*max(0,16-len(a))
        LIVE['estimated_remaining_hours_range']=[round(left/3600*.7,2),round(left/3600*1.5,2)]
    common.write_status(OUT/'logs/progress.json',LIVE)
def split(y,n,seed):return list(StratifiedKFold(n_splits=n,shuffle=True,random_state=seed).split(np.zeros(len(y)),y))
def recipes():
    # Balanced fractional joint grid: all 9 selectors x all 6 counts; three
    # hyperparameters each occur 18 times. Not an exhaustive 162-cell grid.
    return [{'selector':s,'count':k,'config':(i+j)%3} for i,s in enumerate(SELECTORS) for j,k in enumerate(COUNTS)]
def rawscore(m,x):
    if hasattr(m,'decision_function'):return np.asarray(m.decision_function(x),float)
    return logit(np.clip(m.predict_proba(x)[:,1],1e-6,1-1e-6))
def sigmoid(margin,y):return LogisticRegression(C=1.,solver='lbfgs',max_iter=1000,random_state=SEED).fit(np.asarray(margin).reshape(-1,1),y)

class RNAFeatures:
    def __init__(self,genes):self.genes=np.asarray(genes,int)
    def fit(self,x):self.scaler=StandardScaler().fit(x[:,self.genes]);return self
    def transform(self,x):return self.scaler.transform(np.asarray(x)[:,self.genes])

class CalibratedRNAModel:
    def __init__(self,features,estimator,calibrator,recipe,ids):self.features=features;self.estimator=estimator;self.calibrator=calibrator;self.recipe=recipe;self.fit_patients=list(ids)
    def predict_proba(self,x):
        p=self.calibrator.predict_proba(rawscore(self.estimator,self.features.transform(x)).reshape(-1,1))[:,1]
        return np.c_[1-p,p]

def nine_rankings(x,y,valid,seed):
    from boruta import BorutaPy
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    z=StandardScaler().fit_transform(x[:,valid]);raw={};extra={}
    models={
        'Decision Tree':DecisionTreeClassifier(max_depth=4,min_samples_leaf=5,random_state=seed),
        'GBDT':GradientBoostingClassifier(n_estimators=100,max_depth=2,min_samples_leaf=5,learning_rate=.03,random_state=seed),
        'LightGBM':LGBMClassifier(n_estimators=100,num_leaves=7,min_child_samples=10,learning_rate=.03,reg_lambda=10,n_jobs=2,verbosity=-1,random_state=seed,deterministic=True,force_col_wise=True),
        'Logistic Regression':LogisticRegression(C=.1,solver='liblinear',dual=False,max_iter=10000,random_state=seed),
        'Random Forest':RandomForestClassifier(n_estimators=200,max_depth=5,min_samples_leaf=4,n_jobs=2,random_state=seed),
        'XGBoost':XGBClassifier(n_estimators=100,max_depth=2,learning_rate=.03,reg_lambda=10,tree_method='hist',n_jobs=2,random_state=seed)}
    for name,m in models.items():
        publish(rank_selector=name,message='训练子集内拟合九种RNA筛选器')
        m.fit(z,y);raw[name]=np.abs(m.coef_[0]) if hasattr(m,'coef_') else m.feature_importances_
    publish(rank_selector='Boruta',message='训练子集内运行Boruta影子特征筛选')
    boruta=BorutaPy(RandomForestClassifier(n_estimators=120,max_depth=5,min_samples_leaf=4,n_jobs=2,random_state=seed),n_estimators=120,max_iter=30,perc=100,random_state=seed,verbose=0)
    boruta.fit(z,y);hist=np.nanmedian(boruta.importance_history_[1:],axis=0)
    raw['Boruta']=2*boruta.support_.astype(float)+boruta.support_weak_.astype(float)+rankdata(np.nan_to_num(hist),method='average')/len(hist)/2
    extra={'boruta_confirmed_count':int(boruta.support_.sum()),'boruta_tentative_count':int(boruta.support_weak_.sum()),'boruta_note':'Fixed top-k Boruta-informed ranking; top-k membership does not imply confirmed Boruta acceptance'}
    nb=GaussianNB(var_smoothing=.001).fit(z,y)
    ll=-.5*((z-nb.theta_[1])**2/nb.var_[1]-(z-nb.theta_[0])**2/nb.var_[0]);ll0=-.5*(nb.theta_[1]**2/nb.var_[1]-nb.theta_[0]**2/nb.var_[0]);raw['Gaussian Naive Bayes']=np.abs(ll-ll0).mean(0)
    pls=PLSRegression(n_components=2,scale=False).fit(z,y.astype(float));energy=(pls.x_scores_**2).sum(0)*(pls.y_loadings_**2).sum(0);w=pls.x_weights_
    raw['PLS-DA']=np.sqrt(z.shape[1]*((w*w)/(w*w).sum(0)[None,:]*energy[None,:]).sum(1)/max(energy.sum(),1e-20))
    out={};indices=np.flatnonzero(valid)
    for name in SELECTORS:
        imp=np.asarray(raw[name],float);assert np.isfinite(imp).all()
        # Random, training-ID-seeded tie order avoids alphabetic false stability.
        rng=np.random.default_rng(seedof((seed,name)));order=np.lexsort((rng.permutation(len(imp)),-imp))
        allimp=np.full(x.shape[1],np.nan);allimp[valid]=imp;out[name+'__importance']=allimp;out[name+'__order']=indices[order]
    return out,extra

class Data:
    def __init__(self):
        self.patients=pd.read_csv(OUT/'inventory/patient_manifest.csv').sort_values('Patient_ID').reset_index(drop=True)
        t=pd.read_parquet(OUT/'patient_predicted_RNA.parquet');self.ids=self.patients.Patient_ID.to_numpy(str);assert t.index.tolist()==self.ids.tolist()
        self.x=t.to_numpy(float);self.genes=t.columns.to_numpy(str);self.y=self.patients.LNM.to_numpy(int)
        self.fallback=pd.read_csv(OUT/'inventory/RNA_feature_audit.csv').frozen_training_mean_fallback.to_numpy(bool)
        assert self.x.shape==(160,6175) and np.isfinite(self.x).all();self.rank_cache={};self.feature_cache={}
    def ranks(self,tr):
        token=hashlib.sha256(self.ids[tr].tobytes()).hexdigest()[:20]
        if token not in self.rank_cache:
            d=OUT/f'models/training_rankings/{token}';d.mkdir(parents=True,exist_ok=True)
            if (d/'rankings.npz').exists():
                check=json.loads((d/'fit_patients.json').read_text(encoding='utf-8'));assert check['Patient_ID']==self.ids[tr].tolist();assert check['label_sha256']==hashlib.sha256(self.y[tr].tobytes()).hexdigest()
                ranks=dict(np.load(d/'rankings.npz'))
            else:
                valid=(~self.fallback)&(~near_constant_columns(self.x[tr]));assert valid.sum()>=100
                publish(training_node_patients=len(tr),training_node_eligible_RNA=int(valid.sum()))
                ranks,extra=nine_rankings(self.x[tr],self.y[tr],valid,seedof(token))
                np.savez_compressed(d/'rankings.npz',**ranks,eligible=valid)
                dump(d/'fit_patients.json',{'Patient_ID':self.ids[tr].tolist(),'label_sha256':hashlib.sha256(self.y[tr].tobytes()).hexdigest(),'RNA_sha256':hashlib.sha256(self.x[tr].tobytes()).hexdigest(),'eligible_RNA':int(valid.sum()),'seed':seedof(token),'all_nine_selectors_fitted':True,**extra})
            self.rank_cache[token]=ranks
        return self.rank_cache[token]
    def features(self,tr,recipe):
        key=(tuple(tr),recipe['selector'],recipe['count'])
        if key not in self.feature_cache:
            r=self.ranks(tr);g=r[recipe['selector']+'__order'][:recipe['count']];f=RNAFeatures(g).fit(self.x[tr]);self.feature_cache[key]=(f,f.transform(self.x))
        return self.feature_cache[key]

def fit_raw(data,tr,recipe,name,seed):
    f,x=data.features(tr,recipe);m=classifier(name,recipe['config'],seed)
    if name=='SVM':m.set_params(probability=False)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always');m.fit(x[tr],data.y[tr])
    if ws:
        with (OUT/'logs/model_warnings.jsonl').open('a',encoding='utf-8') as o:
            for w in ws:o.write(json.dumps({'time':now(),'classifier':name,'recipe':recipe,'training_patients':len(tr),'type':w.category.__name__,'message':str(w.message)},ensure_ascii=False)+'\n')
    return f,m,x

def calibrated_fit(data,tr,recipe,name,seed,folds):
    margin=np.full(len(tr),np.nan)
    for a,b in folds:
        assert not set(data.ids[tr[a]])&set(data.ids[tr[b]])
        f,m,x=fit_raw(data,tr[a],recipe,name,seed);margin[b]=rawscore(m,x[tr[b]])
    assert np.isfinite(margin).all();cal=sigmoid(margin,data.y[tr]);cp=cal.predict_proba(margin.reshape(-1,1))[:,1]
    f,m,x=fit_raw(data,tr,recipe,name,seed);return CalibratedRNAModel(f,m,cal,recipe,data.ids[tr]),cp,margin

