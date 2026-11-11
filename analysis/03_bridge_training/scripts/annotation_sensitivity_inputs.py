"""Stratify measured QC and aggregate annotated/unannotated RNA without quality assumptions."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd

R=Path(__file__).resolve().parents[2]
O=R/'03_bridge_training/input_cache'
RAW=Path(r'__ESCC_RAW_COSMX_ROOT__')
def main():
    start=time.monotonic()
    fields=['cell_id','fov','cell_ID','nCount_RNA','nFeature_RNA','nCount_negprobes','nCount_falsecode','Area.um2']
    m=pd.read_csv(RAW/'T2415024_metadata_file.csv.gz',usecols=fields)
    a=pd.read_csv(R.parent/'CellMetadata.csv',usecols=['cell_id','celltype'])
    a=a.set_index('cell_id');m['annotation_stratum']=m.cell_id.map(a.celltype).fillna('Unannotated')
    m['has_final_annotation']=m.annotation_stratum.ne('Unannotated')
    m.groupby(['fov','annotation_stratum']).agg(cells=('cell_ID','size'),
       RNA_mean=('nCount_RNA','mean'),RNA_median=('nCount_RNA','median'),
       RNA_features_mean=('nFeature_RNA','mean'),RNA_features_median=('nFeature_RNA','median'),
       negative_mean=('nCount_negprobes','mean'),system_mean=('nCount_falsecode','mean'),
       area_um2_mean=('Area.um2','mean')).to_csv(O/'cell_QC_by_FOV_annotation_stratum.csv')
    m.groupby('annotation_stratum').agg(cells=('cell_ID','size'),RNA_mean=('nCount_RNA','mean'),
       RNA_median=('nCount_RNA','median'),RNA_features_median=('nFeature_RNA','median'),
       negative_mean=('nCount_negprobes','mean'),system_mean=('nCount_falsecode','mean'),
       area_um2_mean=('Area.um2','mean')).to_csv(O/'cell_QC_annotation_summary.csv')
    keys=pd.Index(m.fov.to_numpy(dtype='int64')*1000000+m.cell_ID.to_numpy())
    header=pd.read_csv(RAW/'T2415024_exprMat_file.csv.gz',nrows=0).columns.tolist()
    genes=[x for x in header[2:] if not x.startswith(('Negative','SystemControl'))]
    sums=np.zeros((2,386,len(genes)),dtype='int64');counts=np.zeros((2,386),dtype='int64')
    annotated=m.has_final_annotation.to_numpy();seen=np.zeros(len(m),bool)
    for c in pd.read_csv(RAW/'T2415024_exprMat_file.csv.gz',usecols=['fov','cell_ID']+genes,dtype='int32',chunksize=1500):
        pos=keys.get_indexer(c.fov.to_numpy(dtype='int64')*1000000+c.cell_ID.to_numpy())
        assert (pos>=0).all() and not seen[pos].any();seen[pos]=True
        group=c.fov.to_numpy()-1;flag=annotated[pos].astype(int)
        values=c[genes].to_numpy(copy=False)
        for fid in np.unique(group):
            for k in [0,1]:
                subset=(group==fid)&(flag==k)
                sums[k,fid]+=values[subset].sum(0,dtype='int64');counts[k,fid]+=int(subset.sum())
        if seen.sum()%60000==0:print(f'Annotation sensitivity {seen.sum()}/{len(m)}, {time.monotonic()-start:.0f}s',flush=True)
    assert seen.all()
    raw=pd.read_csv(R/'02_expression/fov_expression_raw.csv.gz').set_index('FOV_ID')
    assert np.array_equal(sums.sum(0),raw[genes].to_numpy())
    np.savez_compressed(O/'annotation_stratified_RNA.npz',genes=np.array(genes),
          annotated_counts=sums[1],unannotated_counts=sums[0],cell_counts=counts,FOV_ID=np.arange(1,387))
    out={'status':'VERIFIED','raw_cells':len(m),'annotated_cells':int(counts[1].sum()),
      'unannotated_cells':int(counts[0].sum()),'all_strata_sum_exactly_to_all_cell_RNA':True,
      'annotation_assumed_quality_label':False,'seconds':time.monotonic()-start}
    (O/'annotation_sensitivity_QA.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out),flush=True)
if __name__=='__main__':main()
