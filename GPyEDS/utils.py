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

def remove_pc_comp(conc_map, comp2remove):
    from sklearn.decomposition import PCA
    dummy_mask = np.ones((conc_map.shape[0], conc_map.shape[1]), dtype = "bool")

    pca = PCA(n_components=conc_map.shape[-1]-1).fit(np.nan_to_num(conc_map[dummy_mask]))
    sc = pca.transform(np.nan_to_num(conc_map[dummy_mask]))

    new_sc = sc.copy()
    new_sc[:,comp2remove] = 0
    nd = pca.inverse_transform(new_sc) + pca.mean_

    conc_map = np.zeros((conc_map.shape[0], conc_map.shape[1],conc_map.shape[2]))
    for i in range(nd.shape[1]):
        conc_map[:,:,i] = get_img(nd[:,i], dummy_mask)


    return conc_map

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
    new_array = np.zeros_like(mask, dtype = values.dtype)
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


def plot_decomp(scores, comps, plot_return = False, elements = None, 
                mask = None, smooth = None, ex_var = None):
    """
    Function to plot the results of the decomposition function. It may be called directly or
    through the clustering function itself.
    
    Input
    -----------
    scores (3D numpy array) - array containing the scores assigned to every pixel
                            with a separate image for every component.
    comps (2D numpy array) - list of component compositions ordered identical to
                            score maps.
    plot_return (bool) - (optional) if True the matplotlib figure and axis objects
                            are returned; default is False
    elements (list) - (optional) x tick labels used for plotting cluster center
                            compositions.
    shape (list) - (optional) if scores matrix passed is 2D, this is the shape used
                            to plot it on; default is to reshape first dimension as 
                            close to square as possible, keeping the second dimension
                            unchanged (not recommended)
    
    Return (if set True)
    -----------
    fig - matplotlib figure object for plot.
    ax - matplotlib axis object for plot.
    """
    
    if len(scores.shape) == 2:
        if mask != None:
            scores = get_img(scores, mask)
        else:
            raise ValueError("If scores passed are 2D array, mask object is required.")
    elif len(scores.shape) == 3:
        pass
    else:
        raise ValueError("Scores array needs to have 1 or 2 dimensions.")
    
    fig, ax = plt.subplots(len(comps), 2, figsize = (12,24))
    

    
    for i in range(len(comps)):
        # vmin = np.min(scores[:,:,i])
        # vmax = np.max(scores[:,:,i])
        # diff = vmax - vmin
        # vmin += 0.25*diff
        # vmax -= 0.25*diff
        vmin, vmax = np.percentile(scores[:,:,i][mask.astype("bool")], [75 ,25])
        #print(scores[:,:,i].shape)
        if smooth is not None:
            img = ax[i][0].imshow(gaussian_filter(scores[:,:,i], mask, smooth), interpolation = "none",vmin = vmin, vmax = vmax)
        else:
            img = ax[i][0].imshow(scores[:,:,i], interpolation = "none",vmin = vmin, vmax = vmax)
        fig.colorbar(img, ax = ax[i][0],fraction=0.02, pad=0.03)
        ax[i][1].bar(range(len(elements)), comps[i], width = 0.5, tick_label = elements)
        ax[i][1].plot([-0.5, len(elements)-0.5], [0, 0], 'k-')
        if ex_var is not None:
            ax[i][1].set_title("Explained Variance Ratio: " + str(np.round(ex_var[i], 4)))

    fig.tight_layout()
    
    if plot_return == True:
        return fig, ax
    else:
        return None
    
# def cluster(data, n_clusters = 2, method = "gmm", data_mask = None,
#             plot = False,plot_return = False, elements = None,
#            df_shape = None):
#     """
#     Function to perform clustering on the dataset passed using the selected clustering
#     algorithm.
    
#     Input
#     ------------
#     data (either 2D or 3D numpy array) - the dataset to perform clustering on.
#     n_clusters (int) - number of clusters to find, default is 2.
#     method (str) - clustering algorithm to be used ["k_means", "gmm"]; default is k_means.
#     data_mask  - 
#     plot (bool) - Make True if results are to be plotted; default is false.
#     plot_return (bool) - optional, if plot=true, make True to return fig and ax objects, default is false.
#     elements (list/array) - optional, used when plotting results only, default is None.
    
#     Return
#     ------------
#     labels (2D numpy array) - assigned labels for each cluster found within the passed dataset. 
#                     Shape is the same as first two dimensions of data if it's 3D, otherwise it's
#                     the shape parameter passed to the function.
#     centers (2D numpy array of shape [n_clusters, n_features]) - list of the centres of clusters
#                     found in the dataset.
#     fig, ax (matplotlib objects (both of length 2)) - only if both plot and plot_return are set True.
#     """
        
#     if isinstance(data, pd.DataFrame):
#         data, data_mask = build_conc_map(data, df_shape)
#     else:
#         pass
    
#     if len(data.shape) == 3:
#         #only keep non 'nan' entries
#         array = data[data_mask.astype('bool')]
#     elif len(data.shape) == 2:
#         #assume it's in the right form
#         array = data
#     else:
#         raise ValueError("Input array needs to have 2 or 3 dimensions or be Pandas dataframe.")
    
#     array, params = feature_normalisation(array, return_params = True)
    
#     start = time.time()
    
        
#     if method.lower() == "gmm":
#         from sklearn.mixture import GaussianMixture
#         #perform GMM
#         gmm = GaussianMixture(n_clusters)
#         labels = gmm.fit_predict(array) + 1
#         centers = gmm.means_

#     elif method.lower() == "k_means":
#         from sklearn.cluster import KMeans
#         #perform k_means clustering
#         kmeans = KMeans(n_clusters=n_clusters, init = 'k-means++').fit(array)
#         labels = kmeans.labels_
#         centers = kmeans.cluster_centers_
        
#     else:
#         raise ValueError("Method " + str(method) + " is not recognised.")
        
#     process_time = time.time() - start    
#     print("Clustering processing time (s): " + str(process_time))
    
#     centers = rescale(centers, params)
#     #convert to 2D image
#     labels = get_img(labels, data_mask)
    
#     if plot == True:
#         fig, ax = plot_cluster(labels, centers, plot_return = True ,elements=elements)
#         if plot_return == True:
#             return labels, centers, fig, ax
#         else:
#             pass
#     else:
#         pass
    
#     return labels, centers

def decompose(data, n_components = 2, method = "pca", tol = 0.05, 
              plot = False, plot_return = False, elements = None, 
             data_mask = None, df_shape = None, smooth = None):
    """
    Function to perform the selected decomposition algorithm on the data passed. Ideal for use in
    the initial data exploration steps.
    
    Input
    ------------
    data (either 2D or 3D numpy array) - the dataset to be decomposed.
    n_components (int) - number of components to be kept after decomposition; default is 2.
    method (str) - choose one of ["pca", "nmf"] as the algorithm to be used; default is pca.
    tol (float) - tolerance value to be used for NMF decomposition; default is 0.05.
    plot (bool) - Make True if results are to be plotted; default is false.
    plot_return (bool) - optional, if plot=true, make True to return fig and ax objects; default is false.
    elements (list/array) - optional, used when plotting results only; default is None.
    data_mask - 
    
    Return
    ------------
    scores (3D numpy array) - scores relating each element in the original dataset to the 
                            components found. Third dimension equal to n_components.
    components (2D numpy array [n_components, n_features]) - Components found to describe the 
                            dataset; ordered by the fraction of dataset variance described by
                            each component.
    fig, ax (single matplotlib objects) - only if both plot and plot_return are set True.
    """
    from sklearn import decomposition
    
    if isinstance(data, pd.DataFrame):
        data, data_mask = build_conc_map(data, df_shape)
    else:
        pass
    
    if len(data.shape) == 3:
        array = data[data_mask.astype('bool')]
    elif len(data.shape) == 2:
        #assume it's in the right form
        array = data
    else:
        raise ValueError("Input array needs to have 2 or 3 dimensions or be Pandas dataframe.")
     
    if data_mask is None:
        data_mask = np.ones((data.shape[0], 1))
    if elements is None:
        elements = [i for i in range(data.shape[-1])]
    
    start = time.time()
    
    if method.lower() == "pca":
        # array, params = feature_normalisation(array, return_params = True)
        # array = np.nan_to_num(array)
        #perform PCA decomposition
        pca = decomposition.PCA(n_components = n_components)
        scores = pca.fit_transform(array)
        components = pca.components_
        ex_var = pca.explained_variance_ratio_
        print(ex_var)
        
    elif method.lower() == "nmf":
        array, params = feature_normalisation(array, return_params = True, mean_norm = False)
        array = np.nan_to_num(array)
        #perform NMF decomposition
        nmf = decomposition.NMF(n_components = n_components, tol = tol)
        scores = nmf.fit_transform(array)
        components = nmf.components_
        ex_var = None
        
    else:
        raise ValueError("Method " + str(method) + " is not recognised.")
    
    process_time = time.time() - start    
    print("Decomposition processing time (s): " + str(process_time))

    
    #components = rescale(components, params)    
    scores_2d = np.zeros((data_mask.shape[0], data_mask.shape[1], n_components))
    for i in range(n_components):
        scores_2d[:,:,i] = get_img(scores[:,i], data_mask)
    
    if plot == True:
        fig, ax = plot_decomp(scores_2d, components, plot_return = True, elements=elements, smooth=smooth,
                               mask=data_mask, ex_var=ex_var)
        if plot_return == True:
            return scores_2d, components, fig, ax
        else:
            pass
    else:
        pass
    
    return scores_2d, components
