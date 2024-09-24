import numpy as np
import matplotlib.pyplot as plt
import scipy

def gkern(l=5, sig=1.):
    """
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def linear_filter(map, mask, range_, type = "mean", sigma = None):

    if type == "mean":
        n = (2*range_ + 1)**2
        filter = [[1/n for _ in range(2*range_+1)] for k in range(2*range_+1)]

    elif type == "gaussian":
        n = 2*range_ + 1
        if sigma is None:
            sigma = range_
        filter = gkern(n, sigma)
    else:
        raise ValueError("Cannot identify method.")

    pmap = np.pad(map, (range_, range_), mode = "constant")
    pmask = np.pad(mask, (range_, range_), mode = "constant")
    
    meanres = scipy.signal.convolve2d(np.multiply(pmap,pmask), np.asarray(filter), boundary = "symm", mode = "same")
    maskres = scipy.signal.convolve2d(pmask, np.asarray(filter), boundary = "symm", mode = "same")

    return np.divide(np.multiply(meanres, pmask), maskres)

def median_filter(map, mask, range_):
    pmap = np.pad(map, (range_, range_), mode = "constant")
    pmask = np.pad(mask, (range_, range_), mode = "constant")

    medianres = np.zeros_like(map)
    k = 2*range_ + 1
    n,m = map.shape
    for i in range(n):
        for j in range(m):

            region = pmap[i:i+k, j:j+k][pmask[i:i+k, j:j+k].astype("bool")]
            if len(region > 0):

                medianres[i, j] += np.median(region)

    medianres[~mask] = 0

    return medianres
