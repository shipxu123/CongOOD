import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as nrms

def call_metrics(y, y_bar, axis=0):
    ssim_res = ssim(y, y_bar, data_range=y.max()-y.min(),channel_axis=axis)
    nrms_res = nrms(y, y_bar,normalization='min-max')
    if np.isnan(ssim_res): 
        ssim_res = None
    elif np.isinf(ssim_res): 
        ssim_res = None
    if np.isnan(nrms_res): 
        nrms_res = None
    elif np.isinf(nrms_res): 
        nrms_res = None
    return ssim_res, nrms_res
