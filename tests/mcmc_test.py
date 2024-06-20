import numpy as np
from GPyEDS.EDS2CHEM import mcmc

def test_draw():
    img = np.zeros((100,100))

    new = mcmc.draw_line(img, 10, 10, 50,50)

    new2 = mcmc.draw_proj_box(img, 10, 10, 50, 50,pstep=5)

def test_simple_mcmc_run():

    dummy = np.random.rand(20)

    y = np.random.rand(20)

    params = [1.,1.]
    pmin = [-10., -10.]
    pmax = [10., 10.]

    mcmc.Simple_MCMC_run(dummy, y, y*0.01, params, pmin, pmax, num_iter = 10)

def test_full_mcmc_run():

    dummy_map = np.random.rand(100,100)

    y = np.random.rand(20)

    params = [1.,1.,1.,1.,1.,1.,1.,1.]
    pmin = [-10., -10.,-10., -10.,-10., -10.,-10., -10.]
    pmax = [10., 10.,10., 10.,10., 10.,10., 10.]

    mcmc.MCMC_run(dummy_map, y, y*0.01, params, pmin, pmax, num_iter = 10)


def test_align_once():
    dummy_map = np.random.rand(100,100)

    pos = np.linspace(1,5,5).astype("int64")
    theta = params = [1.,1.,10.,10.,1.,1.,1.,1.]

    _ = mcmc.align_once(dummy_map, pos, theta)