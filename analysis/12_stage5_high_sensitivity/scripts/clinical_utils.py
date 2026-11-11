"""Binary clinical metrics, score cutoffs and explicitly conditional prevalence projections."""
import numpy as np
from scipy.stats import beta

def choose_cutoff(y,p,target):
    y=np.asarray(y);p=np.asarray(p,dtype=float)
    if len(y)!=len(p) or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():raise ValueError('Invalid labels or probability scores')
    positive=np.sort(p[y==1])
    if not len(positive) or not 0<target<=1:raise ValueError('A positive calibration sample and a target in (0,1] are required')
    required=int(np.ceil(target*len(positive)-1e-12))
    return float(positive[len(positive)-required])

def interval(k,n):
    if n==0:return np.nan,np.nan
    return float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.

def ratio(k,n):return k/n if n else np.nan

def metrics(y,high):
    y=np.asarray(y,dtype=int);high=np.asarray(high,dtype=bool)
    if y.shape!=high.shape or not np.isin(y,[0,1]).all():raise ValueError('Invalid patient decisions')
    tp=int(np.sum((y==1)&high));fn=int(np.sum((y==1)&~high));tn=int(np.sum((y==0)&~high));fp=int(np.sum((y==0)&high));n=len(y)
    r={'patients':n,'positive':tp+fn,'negative':tn+fp,'TP':tp,'FP':fp,'TN':tn,'FN':fn,'high_risk_patients':tp+fp,'low_risk_patients':tn+fn,
       'high_risk_per100':100*ratio(tp+fp,n),'missed_LNM_per100':100*ratio(fn,n),'false_positive_per100':100*ratio(fp,n),
       'true_low_risk_per100':100*ratio(tn,n),'low_risk_per100':100*ratio(tn+fn,n),'prevalence':ratio(tp+fn,n)}
    for name,k,d in [('Sensitivity',tp,tp+fn),('Specificity',tn,tn+fp),('FNR',fn,tp+fn),('NPV',tn,tn+fn),('PPV',tp,tp+fp),('residual_LNM_in_low_risk',fn,tn+fn)]:
        r[name]=ratio(k,d);r[name+'_CI_low'],r[name+'_CI_high']=interval(k,d)
    r['NPV_status']='defined' if tn+fn else 'undefined_no_low_risk_patients'
    return r

def project(sensitivity,specificity,prevalence):
    tp=sensitivity*prevalence;fn=(1-sensitivity)*prevalence;tn=specificity*(1-prevalence);fp=(1-specificity)*(1-prevalence)
    return {'assumed_prevalence':prevalence,'TP_per100':100*tp,'FP_per100':100*fp,'TN_per100':100*tn,'FN_per100':100*fn,
        'high_risk_per100':100*(tp+fp),'low_risk_per100':100*(tn+fn),'missed_LNM_per100':100*fn,
        'PPV':ratio(tp,tp+fp),'NPV':ratio(tn,tn+fn),'residual_LNM_in_low_risk':ratio(fn,tn+fn)}

def net_benefit(sensitivity,specificity,prevalence,preference):
    return sensitivity*prevalence-(1-specificity)*(1-prevalence)*np.asarray(preference)/(1-np.asarray(preference))
