import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import matplotlib.patches as mpatches
import skimage.morphology as morph
import skimage.measure as measure
import skimage.filters as filters


def mean_centre(feature, return_params = False):
    """
    Function to perform mean normalisation on the dataset passed to it.    
    """
    
    
    params = []
    
    norm = np.zeros_like(feature)
    
    if len(feature.shape) == 2:
        for i in range(feature.shape[1]):
            temp_mean = feature[:,i].mean()
            norm[:,i] = feature[:,i] - temp_mean
            params.append(np.asarray([temp_mean]))
    
    elif len(feature.shape) == 1:
        temp_mean = feature[:].mean()
        norm[:] = feature[:] - temp_mean
        params.append(np.asarray([temp_mean]))
        
    else:
        raise ValueError("Feature array must be either 1D or 2D numpy array.")
        
    
    if return_params == True:
        return norm, params
    else:
        return norm