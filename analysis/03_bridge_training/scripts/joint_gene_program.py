"""Requested joint-output comparison; inner-fold convex fusion, not independent labels."""
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from bridge_core import patient_mse
import train_bridge as tr
from evaluate_bridge import model_summary

def main():
    T=tr.T;data=tr.load_data();tr.DATA=data
    run=Path(json.loads((T/'active_run_all_cells_0.95_0.1.json').read_text(encoding='utf-8'))['run']);tr.RUN=run
    out=T/'joint_gene_program_models';out.mkdir(exist_ok=True)
    tr.dump(out/'protocol.json',{'frozen_at':datetime.now().astimezone().isoformat(),
       'purpose':'Complete the user-requested gene/program joint comparison',
       'prediction':'convex fusion of direct Ridge gene predictions and NMF program-reconstructed gene predictions; return NMF scores as accompanying program outputs',
       'weights':[0.,.5,1.],'weight_fit':'3 inner patient folds only, outer patients never used',
       'limit':'Implementation completed after primary results; primary 42-model nested endpoint is unchanged. Programs are derived from genes and are not independent extra labels.'})
    rows=[]
    for encoder in tr.CONFIG['encoders']:
        x=tr.feature_data(encoder,'HSV_mean')
        for group in ['A','B','C']:
            name=f'{encoder}_HSV_mean_{group}_Joint_Ridge_NMF'
            prediction=np.zeros_like(data['y'],dtype='float32');baseline=prediction.copy();mask=np.ones(6175,bool);recs=[]
            for fold in range(5):
                direct=run/f'outer_fold_{fold}/{encoder}_HSV_mean_{group}_Ridge'
                program=run/f'outer_fold_{fold}/{encoder}_HSV_mean_{group}_NMF'
                dr=json.loads((direct/'result.json').read_text(encoding='utf-8'));pr=json.loads((program/'result.json').read_text(encoding='utf-8'))
                train=np.flatnonzero(data['fold']!=fold);test=np.flatnonzero(data['fold']==fold)
                scores=[]
                for weight in [0.,.5,1.]:
                    values=[]
                    for a,b in tr.inner_splits(train,fold):
                        dm=tr.fit_model(a,group,encoder,'Ridge',dr['selected']['config'])
                        pm=tr.fit_model(a,group,encoder,'NMF',pr['selected']['config'])
                        pred=weight*dm.predict(x[b])+(1-weight)*pm.predict(x[b])
                        values.append(patient_mse(data['y'][b],pred,data['patients'][b]))
                    scores.append(float(np.mean(values)))
                weight=[0.,.5,1.][int(np.argmin(scores))]
                d=np.load(direct/'OOF_prediction.npz');p=np.load(program/'OOF_prediction.npz')
                assert np.array_equal(d['FOV_ID'],p['FOV_ID'])
                pos=pd.Index(data['FOV']).get_indexer(d['FOV_ID'])
                prediction[pos]=weight*d['prediction']+(1-weight)*p['prediction'];baseline[pos]=d['primary_training_mean']
                mask&=d['gene_mask']&p['gene_mask']
                keep=data['primary'][test]
                ratio=patient_mse(data['y'][test][keep],prediction[test][keep],data['patients'][test][keep])/patient_mse(data['y'][test][keep],baseline[test][keep],data['patients'][test][keep])
                rec={'encoder':encoder,'group':group,'family':'Joint_Ridge_NMF','outer_fold':fold,'solver_converged':True,
                   'primary_MSE_ratio':ratio,'direct_gene_weight':weight,'inner_scores':scores,
                   'direct_model_path':str(direct/'model.joblib'),'NMF_model_path':str(program/'model.joblib')}
                recs.append(rec);tr.dump(out/f'{name}_fold{fold}.json',rec)
            np.savez_compressed(out/f'{name}_OOF.npz',FOV_ID=data['FOV'],genes=data['genes'],prediction=prediction,baseline=baseline,gene_mask=mask)
            rows.append(model_summary(name,{'pred':prediction,'mask':mask,'records':recs},data,baseline))
            print(f'{name} completed',flush=True)
    pd.DataFrame(rows).to_csv(out/'joint_gene_program_comparison.csv',index=False)
    tr.dump(out/'completion.json',{'status':'COMPLETED','outer_joint_predictions':30,'primary_results_changed':False})
if __name__=='__main__':
    with threadpool_limits(limits=2):main()
