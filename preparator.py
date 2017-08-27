import skvideo.io as skv
import os
from skimage.measure import block_reduce
import numpy as np

def load_reduce_pipeline(path):
    vid = skv.vread(path)
    vid = block_reduce(vid, (1, 5, 5, 1), func=np.max)
    vid = vid[:, :, 100:324, :]
    
    return np.pad(vid, ((0, 0), (4, 4), (0, 0), (0, 0)), "constant", constant_values=(0))