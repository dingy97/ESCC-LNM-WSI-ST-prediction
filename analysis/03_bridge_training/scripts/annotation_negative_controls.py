"""Exact per-probe controls for the annotation-stratified RNA sensitivity."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[2];O=R/'03_bridge_training/input_cache';RAW=Path(r'__ESCC_RAW_COSMX_ROOT__')
a=pd.read_csv(R.parent/'CellMetadata.csv',usecols=['cell_id','fov'])
annotated=pd.Index(a.fov.to_numpy(dtype='int64')*1000000+a.cell_id.str.rsplit('_',n=1).str[-1].astype('int64').to_numpy())
header=pd.read_csv(RAW/'T2415024_exprMat_file.csv.gz',nrows=0).columns.tolist()
probes=[g for g in header if g.startswith('Negative')]
out=np.zeros((2,386,len(probes)),dtype='int64');rows=0;start=time.monotonic()
for c in pd.read_csv(RAW/'T2415024_exprMat_file.csv.gz',usecols=['fov','cell_ID']+probes,dtype='int32',chunksize=10000):
    flag=annotated.get_indexer(c.fov.to_numpy(dtype='int64')*1000000+c.cell_ID.to_numpy())>=0
    group=c.fov.to_numpy()-1;values=c[probes].to_numpy()
    for fid in np.unique(group):
        for k in [0,1]:out[k,fid]+=values[(group==fid)&(flag==bool(k))].sum(0,dtype='int64')
    rows+=len(c)
    if rows%100000==0:print(f'Stratified negative probes {rows}/536232; {time.monotonic()-start:.0f}s',flush=True)
raw=pd.read_csv(R/'02_expression/fov_expression_raw.csv.gz')
assert rows==536232 and np.array_equal(out.sum(0),raw[probes].to_numpy())
np.savez_compressed(O/'annotation_negative_probes.npz',probes=np.array(probes),annotated=out[1],unannotated=out[0])
print('Exact annotation-stratified negative probe counts verified',flush=True)
