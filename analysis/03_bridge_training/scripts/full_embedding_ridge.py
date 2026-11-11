"""Exploratory capacity sensitivity: keep all frozen encoder coordinates, dual Ridge."""
from pathlib import Path
from datetime import datetime
import json
import joblib
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor,cho_solve
from threadpoolctl import threadpool_limits
from bridge_core import patient_weights,patient_mse
from bridge_models import TargetSpace,moments
import train_bridge as tr
from evaluate_bridge import model_summary

def dual_fit(x,y,weights,alpha):
    sw=np.sqrt(weights)
    a=x*sw[:,None]
    coef=cho_solve(cho_factor(a@a.T+alpha*np.eye(len(a)),lower=True),y*sw[:,None])
    return a,coef
def dual_predict(x,fitted):return (x@fitted[0].T)@fitted[1]

class FullRidge:
    def fit(self,x,y,raw,negative,groups,alpha,primary=None):
        w=patient_weights(groups);self.mu,self.sd=moments(x,w)
        self.target=TargetSpace(y,raw,negative,groups);z=self.target.transform(y)
        xx=(x-self.mu)/self.sd
        self.regressor=dual_fit(xx,z,w,alpha*len(xx));self.correction=None
        self.baseline=self.target.mean.copy()
        if primary is not None:
            xp=xx[primary];wp=patient_weights(groups[primary])
            resid=z[primary]-dual_predict(xp,self.regressor)
            self.cm,_=moments(resid,wp);self.xm,_=moments(xp,wp)
            self.correction=dual_fit(xp-self.xm,resid-self.cm,wp,len(xp))
            self.baseline=np.average(y[primary],axis=0,weights=wp)
        return self
    def predict(self,x):
        xx=(x-self.mu)/self.sd;z=dual_predict(xx,self.regressor)
        if self.correction is not None:z+=.5*(self.cm+dual_predict(xx-self.xm,self.correction))
        prediction=self.target.decode(z);prediction[:,~self.target.mask]=self.baseline[~self.target.mask]
        return np.clip(prediction,0,np.log1p(1e6)).astype('float32')

def main():
    R=tr.R;T=tr.T;data=tr.load_data();tr.DATA=data
    out=T/'full_embedding_capacity_sensitivity/v2';out.mkdir(parents=True,exist_ok=True)
    note={'frozen_at':datetime.now().astimezone().isoformat(),
      'status':'Exploratory capacity sensitivity declared after primary results; never replaces the frozen primary nested pipeline',
      'reason':'The common main-analysis input PCA is capped by training patient count. Check whether discarding low-variance encoder coordinates hides RNA signal; motivation is representation capacity, not LNM AUC or training correlation.',
      'feature_transform':'Training-patient mean/SD only, no PCA input compression',
      'alphas_per_training_row':[.1,10.,1000.],'inner_folds':3,'same_inherited_outer_patient_folds':True}
    tr.dump(out/'protocol.json',note)
    results=[];summary=[]
    for encoder in ['ResNet50','Phikon']:
        x=tr.feature_data(encoder,'HSV_mean')
        for group in ['A','B','C']:
            name=f'{encoder}_FULL_{group}_Ridge';pred=np.zeros_like(data['y'],dtype='float32');base=pred.copy();recs=[];mask=np.ones(6175,bool)
            for fold in range(5):
                folder=out/f'fold_{fold}'/name;folder.mkdir(parents=True,exist_ok=True)
                train=np.flatnonzero(data['fold']!=fold);test=np.flatnonzero(data['fold']==fold)
                scores=[]
                for alpha in [.1,10.,1000.]:
                    inner=[]
                    for a,b in tr.inner_splits(train,fold):
                        actual=a[data['primary'][a]] if group=='A' else a
                        model=FullRidge().fit(x[actual],data['y'][actual],data['raw'][actual],data['negative'][actual],data['patients'][actual],alpha,
                              data['primary'][actual] if group=='C' else None)
                        inner.append(patient_mse(data['y'][b],model.predict(x[b]),data['patients'][b]))
                    scores.append(float(np.mean(inner)))
                alpha=[.1,10.,1000.][int(np.argmin(scores))]
                actual=train[data['primary'][train]] if group=='A' else train
                model=FullRidge().fit(x[actual],data['y'][actual],data['raw'][actual],data['negative'][actual],data['patients'][actual],alpha,
                    data['primary'][actual] if group=='C' else None)
                pred[test]=model.predict(x[test]);mask &=model.target.mask
                primary=train[data['primary'][train]]
                baseline=np.average(data['y'][primary],axis=0,weights=patient_weights(data['patients'][primary]));base[test]=baseline
                use=test[data['primary'][test]]
                ratio=patient_mse(data['y'][use],pred[use],data['patients'][use])/patient_mse(data['y'][use],base[use],data['patients'][use])
                joblib.dump(model,folder/'model.joblib',compress=3)
                np.savez_compressed(folder/'OOF_prediction.npz',FOV_ID=data['FOV'][test],prediction=pred[test],gene_mask=model.target.mask)
                record={'encoder':encoder,'family':'Full_embedding_Ridge','group':group,'outer_fold':fold,
                  'selected_alpha_per_row':alpha,'inner_primary_MSEs':scores,'primary_MSE_ratio':ratio,'solver_converged':True}
                tr.dump(folder/'result.json',record);recs.append(record)
                print(f'{name} fold{fold}: {ratio:.4f}',flush=True)
            np.savez_compressed(out/f'{name}_OOF.npz',FOV_ID=data['FOV'],genes=data['genes'],prediction=pred,baseline=base,gene_mask=mask)
            entry={'pred':pred,'mask':mask,'records':recs};summary.append(model_summary(name,entry,data,base));results+=recs
    pd.DataFrame(summary).to_csv(out/'full_embedding_Ridge_comparison.csv',index=False)
    tr.dump(out/'completion.json',{'status':'COMPLETED','outer_fits':len(results),'primary_results_unchanged':True})
if __name__=='__main__':
    with threadpool_limits(limits=2):main()
