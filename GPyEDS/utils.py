# -*- coding: utf-8 -*-
"""
@author: norbert
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import matplotlib.patches as mpatches
import skimage.morphology as morph
import skimage.measure as measure
import skimage.filters as filters

def split_at(string, char, n):
    """
    Splits string into two at the nth occurence of the character specified.
    
    Input
    ------------------------------------
    string - string to be split into two
    char - character to split at
    n - the occurrence of character at which splitting should occur.
    
    Returns
    -------------------------------------
    The two ends of the string split at the specified position.
    """
    words = string.split(char)
    return char.join(words[:n]), char.join(words[n:])


def get_img(values, mask):
    """
    Function to transform cluster or decomposition results into a displayable image.
    It adds 'nan' to all pixels where no data exists.

    Parameters
    ----------
    values : 1D ndarray
        List of values to be transformed into 2D ndarray.
    mask : 2D ndarray (bianry mask)
        Mask showing location of values in dataset.

    Returns
    -------
    new_array : 2D ndarray
        Transformed image showing the values passed in their respective locations.

    """    
    new_array = np.zeros_like(mask)
    new_array[:] = np.nan

    np.place(new_array, mask.astype('bool'), values)
    
    return new_array

def list2stack(array_list):
    """
    Function to stack lists of 2D numpy arrays to a 3D stack.

    Parameters
    ----------
    array_list : list of 2D ndarray
        list of 2D ndarray objects to stack into 3D ndarray

    Returns
    -------
    stack : 3D ndarray
        Stacked ndarray object.

    """
    
    if isinstance(array_list, list) == False:
        raise ValueError("Object passed must be a list.")
    elif len(array_list[0].shape) != 2:
        raise ValueError("List object must contain 2D arrays.")
    else:
        pass

    stack = np.empty((array_list[0].shape[0], array_list[0].shape[1], len(array_list)))
    
    for i in range(len(array_list)):
        stack[:,:,i] = array_list[i]
        
    if len(array_list) == 1:
        stack = stack[:,:,0]
    else:
        pass
    
    return stack


def stack2list(stack):
    """
    Opposite/reverse of list2stack function.

    Parameters
    ----------
    stack : 3D ndarray
        stacked array of 2D maps to be broken up into list of 2D ndarray.

    Returns
    -------
    array_list : list of 2D ndarray
        broken up list of 2D ndarray maps.

    """
    
    if len(stack.shape) != 3:
        raise ValueError("Stack object passed must be 3D numpy array.")
    else:
        pass    
    
    array_list = [stack[:,:,i] for i in range(stack.shape[2])]
    
    return array_list


def gauss_check(item):
    """
    Checks input parameters of Gaussian filter function for appropriate
    types. Allows for the handling of multipl types of parameters eg. lists
    and ndarray objects both.

    Parameters
    ----------
    item : ndarray or list of ndarrays
        Original input parameter to be checked.
        
    Returns
    -------
    item_list : list of 2D ndarray objects
        list of 2D ndarray objects to be used for gaussian filter separately.

    """
    
    if isinstance(item, list) == True:
        item_list = item
    elif isinstance(item, np.ndarray) == True:
        if len(item.shape) == 3:
            item_list = stack2list(item)
        elif len(item.shape) == 2:
            item_list = [item]
        else:
            raise ValueError("Stack object passed must be 2D or 3D numpy array.")
    else:
        raise ValueError("Concentration map passed must be a list or 2/3D numpy array.")
        
    return item_list
    
    
def gaussian_filter(conc, mask, std = 5, list_return = False):
    """
    Wrapper for the skimage implementation of the Gaussian filter function. 
    This implementation takes into account the different phases present and limits
    the smoothing to each phase only - avoiding the creation of artifical 'mixels'
    upon smoothing. Useful for noisy datasets, but beware of drawbacks/limitations.

    Parameters
    ----------
    conc : 2/3D ndarray or list of 2D ndarray
        Concentration map(s) to smoothe.
    mask : binary mask 2/3D ndarray or list of 2D ndarrays.
        Binary mask showing the positions of the phases present in the dataset.
    std : int, optional
        Standard deviation of Gaussian kernel used for smoothing. The default
        is 5.
    list_return : bool, optional
        If True, filtered maps are returned as a list. The default is False so
        result is returned as a 3D stack of ndarray.

    Returns
    -------
    filtered_maps: 3D ndarray stack or list of 2D ndarray
        Result of the smoothing operations.

    """
    
    
    conc_list = gauss_check(np.nan_to_num(conc))
    mask_list = gauss_check(np.nan_to_num(mask))
    
    len_check = len(conc_list) == len(mask_list)
    if len_check == False:
        raise ValueError("Input parameters do not match in dimension.")
    else:
        pass
    
    filtered_maps = []
    
    for i in range(len(conc_list)):
        gauss_conc = filters.gaussian(np.multiply(conc_list[i], mask_list[i]), std, truncate = 10)
        gauss_mask = filters.gaussian(mask_list[i], std, truncate = 10)
        
        gauss_conc = gauss_conc[mask_list[i].astype('bool')]
        gauss_mask = gauss_mask[mask_list[i].astype('bool')]
        corrected = gauss_conc / gauss_mask
                
        filtered_maps.append(get_img(corrected, mask_list[i]))
        
    if list_return == True:
        return filtered_maps
    else:
        return list2stack(filtered_maps)


def feature_normalisation(feature, return_params = False, mean_norm = True):
    """
    Function to perform mean normalisation on the dataset passed to it.
    
    Input
    ----------
    feature (numpy array) - features to be normalised
    return_params (boolean, optional) - set True if parameters used for mean normalisation
                            are to be returned for each feature
                            
    Returns
    ----------
    norm (numpy array) - mean normalised features
    params (list of numpy arrays) - only returned if set to True above; list of parameters
                            used for the mean normalisation as derived from the features
                            (ie. mean, min and max).
    
    """
    
    
    params = []
    
    norm = np.zeros_like(feature)
    
    if len(feature.shape) == 2:
        for i in range(feature.shape[1]):
            if mean_norm == True:
                temp_mean = feature[:,i].mean()
            elif mean_norm == False:
                temp_mean = 0
            else:
                raise ValueError("Mean_norm must be boolean")
            norm[:,i] = (feature[:,i] - temp_mean) / (feature[:,i].max() - feature[:,i].min())
            params.append(np.asarray([temp_mean,feature[:,i].min(),feature[:,i].max()]))
    
    elif len(feature.shape) == 1:
        if mean_norm == True:
            temp_mean = feature[:].mean()
        elif mean_norm == False:
                temp_mean = 0
        else:
            raise ValueError("Mean_norm must be boolean")
        norm[:] = (feature[:] - temp_mean) / (feature.max() - feature.min())
        params.append(np.asarray([temp_mean,feature[:].min(),feature[:].max()]))
        
    else:
        raise ValueError("Feature array must be either 1D or 2D numpy array.")
        
    
    if return_params == True:
        return norm, params
    else:
        return norm

def get_masks(label_array, values = None, return_list = False):
    """
    Creates a list of masks from a label array passed to it.

    Parameters
    ----------
    label_array : 2D numpy array
        Array containing phase labels assigned to each pixel.
    values : list of ints/labels, optional
        The labels to be masked from the label array. The default is None.

    Returns
    -------
    masks : list of 2D numpy arrays
        Colection of phase masks generated.

    """
    
    if values == None:
        values = np.unique(label_array)
        a = True if True in np.isnan(np.array(values)) else False
        if a == True:
            values = [x for x in values if str(x) != 'nan']
        else:
            pass
    elif isinstance(values, int) == True:
        values = [values]
    else:
        pass
    
    masks = np.empty((label_array.shape[0], label_array.shape[1], len(values)))
    masks[:] = np.nan
    
    for i in range(len(values)):
        temp_mask = masks[:,:,i]
        temp_mask[label_array == values[i]] = 1
        
    if len(values) == 1:
        masks = masks[:,:,0]
        
    if return_list == True:
        return stack2list(masks)
    else:  
        return masks


def build_conc_map(data, shape=None,):
    """
    Build 3D numpy conc_map from pandas dataframe.
    
    Input
    -----------
    data (pd dataframe)     
        pandas dataframe to be transformed to numpy data matrix.
    shape (list) (optional) 
        desired shape of the resulting array; if not given
        one will be generated using the data and shape of dataframe.
                        
    Return
    -----------
    conc_map (3D numpy array) 
        the resulting data matrix of shape either given or
        calculated.
    data_mask (2D numpy array) 
        mask showing where data exists in conc_map (binary mask)
    """
    
    if isinstance(data, pd.DataFrame):
        pass
    else:
        raise ValueError("Data is not pandas dataframe.")
    
    if shape == None:
        x_max = data['X'].max() +1
        y_max = data['Y'].max() +1
        n_features = len(data.columns) - 2 #need to substract x,y
        shape = [x_max, y_max, n_features]
    else:
        pass
    
    conc_map = np.zeros(shape)
    conc_map.fill(np.nan)
    data_mask = np.zeros((shape[0], shape[1]))
    k = 1
    length = len(data)
    
    print("Starting to build data-cube.")
    for i in range(length):
        x = data.loc[i, 'X']
        y = data.loc[i, 'Y']
        conc_map[x,y] = data.iloc[i,2:].to_numpy()
        data_mask[x,y] += 1

        if int((i/length)*10) == k:
            print("Progress: " + str(k*10) + "%")
            k += 1
        else:
            pass
    print("Progress: " + str(100)+ "%")

    return conc_map, data_mask

