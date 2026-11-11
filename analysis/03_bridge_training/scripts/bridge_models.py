"""Training-only weighted transforms and small regularized FOV regressors."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import warnings
import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import ConvergenceWarning
from bridge_core import patient_weights,weighted_ridge

SEED=20261104

def moments(x,w):
    mean=np.average(x,axis=0,weights=w)
    scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=w))
    return mean,np.maximum(scale,1e-6)

class TargetSpace:
    def __init__(self,y,raw,negative,groups,quantile=.95,fraction=.1):
        self.weights=patient_weights(groups);self.mean,self.scale=moments(y,self.weights)
        above=raw>np.quantile(negative,quantile,axis=1)[:,None]
        support=np.stack([above[groups==g].mean(0)>=.5 for g in np.unique(groups)]).mean(0)
        variable=np.average((y-self.mean)**2,axis=0,weights=self.weights)>1e-12
        self.mask=(support>=fraction)&variable
        self.support=support;self.variable=variable
        self.reason=np.where(~variable,'zero_training_variance',np.where(support<fraction,'below_training_negative_background_support','included'))
        if self.mask.sum()==0:raise ValueError('No genes pass training-only QC')
    def transform(self,y):return (y[:,self.mask]-self.mean[self.mask])/self.scale[self.mask]
    def decode(self,z):
        out=np.tile(self.mean,(len(z),1));out[:,self.mask]+=z*self.scale[self.mask]
        return out

class FeatureSpace:
    def __init__(self,x,groups):
        w=patient_weights(groups);self.mean,self.scale=moments(x,w)
        rank=min(32,len(np.unique(groups))-1,len(x)-1,x.shape[1])
        z=(x-self.mean)/self.scale
        _,_,self.basis=randomized_svd(z*np.sqrt(w)[:,None],n_components=rank,n_iter=3,random_state=SEED)
        scores=z@self.basis.T
        self.score_mean,self.score_scale=moments(scores,w)
    def transform(self,x):return (((x-self.mean)/self.scale)@self.basis.T-self.score_mean)/self.score_scale

class ProgramSpace:
    def __init__(self,kind,rank,target,y,groups):
        self.kind=kind;self.target=target;self.rank=rank;self.n_iter=None
        w=patient_weights(groups);z=target.transform(y)
        if kind=='PCA':
            _,_,self.basis=randomized_svd(z*np.sqrt(w)[:,None],n_components=rank,n_iter=4,random_state=SEED)
        elif kind=='NMF':
            self.nmf=NMF(n_components=rank,init='nndsvda',solver='cd',max_iter=250,tol=1e-3,random_state=SEED)
            with warnings.catch_warnings(record=True) as messages:
                warnings.simplefilter('always',ConvergenceWarning)
                self.nmf.fit(y[:,target.mask]*np.sqrt(w)[:,None])
                self.converged=not any(issubclass(m.category,ConvergenceWarning) for m in messages)
            self.n_iter=int(self.nmf.n_iter_);self.basis=self.nmf.components_
        elif kind=='Coexpression':
            means=np.stack([z[groups==g].mean(0) for g in np.unique(groups)])
            profile=means.T;profile=profile-profile.mean(1,keepdims=True)
            profile=profile/np.maximum(np.linalg.norm(profile,axis=1,keepdims=True),1e-9)
            self.labels=KMeans(n_clusters=rank,n_init=5,random_state=SEED).fit_predict(profile)
            self.basis=np.zeros((rank,z.shape[1]))
            for k in range(rank):
                select=self.labels==k
                if select.sum()==1:loading=np.ones((1,1))
                else:_,_,loading=randomized_svd(z[:,select]*np.sqrt(w)[:,None],n_components=1,n_iter=3,random_state=SEED)
                if loading.sum()<0:loading=-loading
                self.basis[k,select]=loading[0]
        else:raise ValueError(kind)
        scores=self.true_scores(y)
        self.mean,self.scale=moments(scores,w)
    def true_scores(self,y):
        if self.kind=='NMF':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',ConvergenceWarning)
                return self.nmf.transform(np.maximum(y[:,self.target.mask],0))
        return self.target.transform(y)@self.basis.T
    def transform(self,y):return (self.true_scores(y)-self.mean)/self.scale
    def decode(self,z):
        score=z*self.scale+self.mean
        out=np.tile(self.target.mean,(len(z),1))
        centered=(score-self.mean)@self.basis
        if self.kind!='NMF':centered*=self.target.scale[self.target.mask]
        out[:,self.target.mask]+=centered
        return out
    def full_loadings(self):
        out=np.zeros((self.rank,len(self.target.mean)))
        out[:,self.target.mask]=self.basis if self.kind=='NMF' else self.basis*self.target.scale[self.target.mask]
        return out

def neural_fit(x,y,w,config):
    import torch
    torch.set_num_threads(2);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    net=torch.nn.Sequential(torch.nn.Linear(x.shape[1],config['width']),torch.nn.ReLU(),
         torch.nn.Dropout(.2),torch.nn.Linear(config['width'],y.shape[1])).to(device)
    xx=torch.tensor(x,dtype=torch.float32,device=device);yy=torch.tensor(y,dtype=torch.float32,device=device)
    ww=torch.tensor(w/w.sum(),dtype=torch.float32,device=device)
    opt=torch.optim.AdamW(net.parameters(),lr=.01,weight_decay=config['decay'])
    loss_value=None
    for epoch in range(config['epochs']):
        opt.zero_grad(set_to_none=True);pred=net(xx)
        loss=(((pred-yy)**2).mean(1)*ww).sum()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite MLP loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step();loss_value=float(loss.detach())
    layers=[(net[i].weight.detach().cpu().numpy().T.copy(),net[i].bias.detach().cpu().numpy().copy()) for i in [0,3]]
    del net,xx,yy,ww,opt;torch.cuda.empty_cache()
    return layers,loss_value

class BridgeEstimator:
    def __init__(self,family,config,target,feature,program=None):
        self.family=family;self.config=config;self.target=target;self.feature=feature;self.program=program
        self.converged=getattr(program,'converged',True);self.correction=None
    def target_transform(self,y):return self.program.transform(y) if self.program else self.target.transform(y)
    def fit(self,x,y,groups):
        w=patient_weights(groups);xx=self.feature.transform(x);z=self.target_transform(y)
        if self.family in ['Ridge','PCA','NMF','Coexpression']:
            self.coef=weighted_ridge(xx,z,w,self.config.get('alpha',.1)*len(xx))
        elif self.family=='ElasticNet':
            self.model=MultiTaskElasticNet(alpha=self.config['alpha'],l1_ratio=.5,fit_intercept=False,
                max_iter=500,tol=1e-3,selection='cyclic',random_state=SEED)
            with warnings.catch_warnings(record=True) as messages:
                warnings.simplefilter('always',ConvergenceWarning)
                self.model.fit(xx*np.sqrt(w)[:,None],z*np.sqrt(w)[:,None])
                self.converged=not any(issubclass(m.category,ConvergenceWarning) for m in messages)
        elif self.family=='PLS':
            # Symmetric row augmentation prevents sklearn's unweighted centering from
            # changing the already patient-weighted center. Both cross-products are preserved.
            a=xx*np.sqrt(w)[:,None];b=z*np.sqrt(w)[:,None]
            self.model=PLSRegression(n_components=min(self.config['rank'],xx.shape[1]),scale=False,max_iter=300,tol=1e-5)
            with warnings.catch_warnings(record=True) as messages:
                warnings.simplefilter('always',ConvergenceWarning)
                self.model.fit(np.concatenate([a,-a]),np.concatenate([b,-b]))
                self.converged=not any(issubclass(m.category,ConvergenceWarning) for m in messages)
        elif self.family=='MLP':self.layers,self.train_loss=neural_fit(xx,z,w,self.config)
        else:raise ValueError(self.family)
        self.fit_patients=sorted(np.unique(groups).tolist())
        self.baseline=self.target.mean.copy()
        return self
    def latent_predict(self,xx):
        if hasattr(self,'coef'):out=xx@self.coef
        elif self.family=='MLP':out=np.maximum(xx@self.layers[0][0]+self.layers[0][1],0)@self.layers[1][0]+self.layers[1][1]
        else:out=self.model.predict(xx)
        if self.correction:
            c=self.correction;out=out+.5*(c['mean']+(xx-c['xmean'])@c['coef'])
        return out
    def specialize(self,x,y,groups):
        xx=self.feature.transform(x);w=patient_weights(groups)
        residual=self.target_transform(y)-self.latent_predict(xx)
        xm,_=moments(xx,w);ym,_=moments(residual,w)
        self.correction={'mean':ym,'xmean':xm,'coef':weighted_ridge(xx-xm,residual-ym,w,1.*len(xx))}
        self.baseline=np.average(y,axis=0,weights=w)
        self.primary_fit_patients=sorted(np.unique(groups).tolist())
        return self
    def predict(self,x):
        z=self.latent_predict(self.feature.transform(x))
        out=self.program.decode(z) if self.program else self.target.decode(z)
        out[:,~self.target.mask]=self.baseline[~self.target.mask]
        return np.clip(out,0,np.log1p(1e6)).astype('float32')
