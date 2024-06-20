from GPyEDS import GPAM
import numpy as np 

def test_GPAM():

    dummy = np.ones((100,7))

    m1 = GPAM.create_two_layer_GPAM_from_data(dummy)
    m2 = GPAM.create_two_layer_GPAM_from_scratch(7, 100)

    _ = GPAM.model_inference(dummy, m1.layers[1])