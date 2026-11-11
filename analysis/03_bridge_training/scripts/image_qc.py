"""Per-image optical background correction; no fitted cross-patient parameters."""
import numpy as np
def relative_od_fraction(rgb):
    white=np.percentile(rgb,95,axis=(0,1)).astype('float32')
    ratio=np.clip((rgb.astype('float32')+1)/(white[None,None,:]+1),1/256.,1.)
    od=-np.log(ratio)
    return float((od.mean(2)>.08).mean())
