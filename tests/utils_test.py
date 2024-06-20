from GPyEDS import utils
import numpy as np
import pandas as pd

def test_split():

    word = "norm"
    r1, r2 = utils.split_at(word, "o", 0)

def test_get_img():

    mask = np.ones((10,10))
    rand = np.random.rand(100)

    img = utils.get_img(rand, mask)

def test_stacking():
    l = [np.ones((10, 10)), np.ones((10,10))]

    s = utils.list2stack(l)
    l2 = utils.stack2list(s)

def test_gauss_filter():
    mask = np.ones((10,10))
    rand = np.random.rand(10,10)

    f = utils.gaussian_filter(rand, mask)

def test_feature_norm():

    dummy = np.random.rand(100, 7)

    norm, params = utils.feature_normalisation(dummy, True)

def test_get_masks():
    dummy = np.random.randint(5, size = (100,100))

    masks = utils.get_masks(dummy)

def test_build_conc():
    x = np.linspace(0,9,10)
    xx,yy = np.meshgrid(x,x)
    r = np.random.rand(100)
    df = pd.DataFrame(data = {"X": xx.ravel().astype("int64"), "Y": yy.ravel().astype("int64"), "val": r})

    _ = utils.build_conc_map(df)