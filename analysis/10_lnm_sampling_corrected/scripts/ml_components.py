"""Fold-fitted preprocessing, nine RNA rankers and twelve classifiers."""
from common import SEED
import numpy as np
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_decomposition import PLSRegression

SELECTORS=['Boruta','Decision Tree','GBDT','Gaussian Naive Bayes','LightGBM','Logistic Regression','PLS-DA','Random Forest','XGBoost']
CLASSIFIERS=['Logistic Regression','SVM','Random Forest','MLP','LightGBM','KNN','GBDT','Gaussian Naive Bayes','Extra Trees','CatBoost','AdaBoost','XGBoost']
COUNTS=[10,30,100,300,1000,6175]

def near_constant_columns(x):
    return np.ptp(x,axis=0)<=8*np.finfo('float32').eps*np.maximum(1,np.abs(x).mean(0))

class FittedFeatures:
    def __init__(self, kind, gene_indices=None, rank=32):
        self.kind=kind; self.gene_indices=gene_indices; self.rank=rank

    def columns(self, x):
        if self.kind=='B': return np.asarray(x)[:,self.gene_indices]
        if self.kind=='D': return np.c_[x[:,:2816],x[:,2816+np.asarray(self.gene_indices)],x[:,-4:]]
        return np.asarray(x)

    def fit(self, x):
        z=self.columns(x); self.scaler=StandardScaler().fit(z)
        self.near_constant=near_constant_columns(z)
        self.weights=np.ones(z.shape[1])
        if self.kind=='D':
            g=len(self.gene_indices)
            self.weights[:2816]=1/np.sqrt(2816);self.weights[2816:2816+g]=1/np.sqrt(g);self.weights[-4:]=.5
        z=self.scaler.transform(z)*self.weights
        z[:,self.near_constant]=0
        self.pca=None
        if z.shape[1]>self.rank:
            self.pca=PCA(n_components=min(self.rank,len(z)-2),svd_solver='full').fit(z)
            z=self.pca.transform(z)
        self.output_scaler=StandardScaler().fit(z)
        self.n_fit=len(x)
        return self

    def transform(self, x):
        z=self.scaler.transform(self.columns(x))*self.weights
        z[:,self.near_constant]=0
        if self.pca is not None: z=self.pca.transform(z)
        return self.output_scaler.transform(z)

class PatientModel:
    def __init__(self, transform, classifier):
        self.transform=transform;self.classifier=classifier
    def predict_proba(self,x):
        return self.classifier.predict_proba(self.transform.transform(x))

def classifier(name, index=0, seed=SEED):
    if name=='Logistic Regression': return LogisticRegression(C=[.01,.1,1][index],solver='liblinear',random_state=seed,max_iter=1000)
    if name=='SVM':
        from sklearn.svm import SVC
        return SVC(C=[.1,1,10][index],kernel='rbf',gamma='scale',probability=True,random_state=seed)
    if name in ['Random Forest','Extra Trees']:
        cls=RandomForestClassifier if name=='Random Forest' else ExtraTreesClassifier
        return cls(n_estimators=200,max_depth=[3,6,None][index],min_samples_leaf=4,random_state=seed,n_jobs=2)
    if name=='MLP':
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=([16],[32],[64])[index],alpha=[1,.1,.01][index],max_iter=600,random_state=seed,early_stopping=False)
    if name=='LightGBM':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=150,num_leaves=[3,7,15][index],min_child_samples=10,learning_rate=.03,reg_lambda=10,
            random_state=seed,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True)
    if name=='KNN':
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=[3,7,15][index],weights='distance',n_jobs=2)
    if name=='GBDT': return GradientBoostingClassifier(n_estimators=150,max_depth=[1,2,3][index],learning_rate=.03,min_samples_leaf=5,random_state=seed)
    if name=='Gaussian Naive Bayes': return GaussianNB(var_smoothing=[1e-8,.001,.1][index])
    if name=='CatBoost':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=200,depth=[3,4,5][index],learning_rate=.03,l2_leaf_reg=10,thread_count=2,
            random_seed=seed,verbose=False,allow_writing_files=False)
    if name=='AdaBoost': return AdaBoostClassifier(n_estimators=150,learning_rate=[.05,.1,.5][index],random_state=seed)
    if name=='XGBoost':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=150,max_depth=[1,2,3][index],learning_rate=.03,reg_lambda=10,subsample=1,
            colsample_bytree=1,n_jobs=2,random_state=seed,tree_method='hist',eval_metric='logloss')
    raise ValueError(name)

def _ranks(importance):
    importance=np.nan_to_num(importance,nan=0,posinf=0,neginf=0)
    positive=importance>0
    result=np.zeros(len(importance))
    if positive.any():result[positive]=rankdata(importance[positive],method='average')/positive.sum()
    return result

def rna_rankings(x,y,seed=SEED):
    """No validation outcomes or samples are accepted by this function."""
    from boruta import BorutaPy
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    valid=~near_constant_columns(x)
    x=StandardScaler().fit_transform(x)
    x[:,~valid]=0
    xv=x[:,valid]; records={}; raw={}
    models={
        'Decision Tree':DecisionTreeClassifier(max_depth=4,min_samples_leaf=5,random_state=seed),
        'GBDT':GradientBoostingClassifier(n_estimators=100,max_depth=2,min_samples_leaf=5,learning_rate=.03,random_state=seed),
        'LightGBM':LGBMClassifier(n_estimators=100,num_leaves=7,min_child_samples=10,learning_rate=.03,reg_lambda=10,n_jobs=2,verbosity=-1,random_state=seed,deterministic=True,force_col_wise=True),
        'Logistic Regression':LogisticRegression(C=.1,solver='liblinear',dual=True,max_iter=1000,random_state=seed),
        'Random Forest':RandomForestClassifier(n_estimators=200,max_depth=5,min_samples_leaf=4,n_jobs=2,random_state=seed),
        'XGBoost':XGBClassifier(n_estimators=100,max_depth=2,learning_rate=.03,reg_lambda=10,tree_method='hist',n_jobs=2,random_state=seed)}
    for name,model in models.items():
        model.fit(xv,y)
        imp=np.abs(model.coef_[0]) if hasattr(model,'coef_') else model.feature_importances_
        raw[name]=imp
    boruta=BorutaPy(RandomForestClassifier(n_estimators=120,max_depth=5,min_samples_leaf=4,n_jobs=2,random_state=seed),
        n_estimators=120,max_iter=30,perc=100,random_state=seed,verbose=0)
    boruta.fit(xv,y)
    history=np.nanmedian(boruta.importance_history_[1:],axis=0)
    raw['Boruta']=2*boruta.support_.astype(float)+boruta.support_weak_.astype(float)+_ranks(history)/2
    # Gaussian NB: exact additive log-likelihood-ratio contributions, relative to training-mean substitution.
    nb=GaussianNB(var_smoothing=.001).fit(xv,y)
    ll=-.5*((xv-nb.theta_[1])**2/nb.var_[1]-(xv-nb.theta_[0])**2/nb.var_[0])
    ll_mean=-.5*(nb.theta_[1]**2/nb.var_[1]-nb.theta_[0]**2/nb.var_[0])
    raw['Gaussian Naive Bayes']=np.abs(ll-ll_mean).mean(0)
    pls=PLSRegression(n_components=min(2,len(xv)-2),scale=False).fit(xv,np.asarray(y,dtype=float))
    t,w,q=pls.x_scores_,pls.x_weights_,pls.y_loadings_
    energy=(t*t).sum(0)*(q*q).sum(0)
    raw['PLS-DA']=np.sqrt(xv.shape[1]*((w*w)/(w*w).sum(0)[None,:]*energy[None,:]).sum(1)/max(energy.sum(),1e-20))
    for name in SELECTORS:
        arr=np.zeros(x.shape[1]);arr[valid]=_ranks(raw[name]);records[name]=arr
    # Ten patient bootstrap LR fits supply train-only selection stability; consensus is not based on outer AUC.
    rng=np.random.default_rng(seed); frequency=np.zeros(x.shape[1]); bootstrap_rank=np.zeros(x.shape[1])
    for b in range(10):
        idx=np.r_[rng.choice(np.flatnonzero(y==0),sum(y==0)),rng.choice(np.flatnonzero(y==1),sum(y==1))]
        model=LogisticRegression(C=.1,solver='liblinear',dual=True,max_iter=1000,random_state=seed+b).fit(x[idx],np.asarray(y)[idx])
        rank=_ranks(np.abs(model.coef_[0])); bootstrap_rank+=rank/10
        frequency[np.argsort(-rank,kind='stable')[:100]]+=.1
    mean_rank=np.mean([records[n] for n in SELECTORS],axis=0)
    records['Consensus']=.5*mean_rank+.25*bootstrap_rank+.25*frequency
    records['bootstrap_top100_frequency']=frequency
    return records

def top_indices(rank,count):
    return np.argsort(-rank,kind='stable')[:count]
