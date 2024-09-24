from GPyEDS import spatial_filters
import numpy as np

def create_data():

    cmap = np.zeros((10,10), dtype="float64")

    for i in range(10):
        cmap[:,i] = 8-i -0.5
        cmap[i, :] -= i/4

    cmap[cmap > 3.5] = 3.5
    cmap[cmap < 0.51] = 0.5
    map = cmap.copy()
    cmap += np.random.randn(10,10)/5
    mask = map > 0.5
    return cmap, mask

def test_median():
    cmap, mask = create_data()
    res = spatial_filters.median_filter(cmap, mask, 1)

def test_mean():
    cmap, mask = create_data()
    res = spatial_filters.linear_filter(cmap, mask, 1)

def test_gaussian():
    cmap, mask = create_data()
    res = spatial_filters.linear_filter(cmap, mask, 1, type = "gaussian")