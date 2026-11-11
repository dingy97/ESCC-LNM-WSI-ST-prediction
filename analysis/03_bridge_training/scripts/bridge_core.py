"""Small numerical primitives; FOV training units and equal patient inference weights."""
import numpy as np

def patient_weights(groups):
    _,inverse,counts=np.unique(groups,return_inverse=True,return_counts=True)
    weights=1./counts[inverse]
    return weights/weights.mean()

def patient_mse(actual,predicted,groups):
    error=np.mean((actual-predicted)**2,axis=1)
    return float(np.average(error,weights=patient_weights(groups)))

def pool_features(features,weights):
    weights=np.asarray(weights,dtype=float)
    if not np.isfinite(weights).all() or weights.sum()<=0:raise ValueError('No valid image weight')
    return np.average(features,axis=0,weights=weights)

def weighted_ridge(x,y,weights,alpha):
    sw=np.sqrt(weights)[:,None]
    a=np.asarray(x,dtype='float64')*sw
    b=np.asarray(y,dtype='float64')*sw
    return np.linalg.solve(a.T@a+alpha*np.eye(a.shape[1]),a.T@b)

def safe_correlations(actual,predicted):
    a=actual-actual.mean(0);b=predicted-predicted.mean(0)
    denom=np.sqrt((a*a).sum(0)*(b*b).sum(0))
    return np.divide((a*b).sum(0),denom,out=np.full(a.shape[1],np.nan),where=denom>1e-12)
