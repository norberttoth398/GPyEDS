import gpflow
import gpflux
import tensorflow as tf
import matplotlib.pylab as plt
import numpy as np
from pathlib import Path
import os

def create_two_layer_GPAM_from_data(input_data, num_inducing = 50, return_layers = False, n_latent = 2):
    """_summary_

    Args:
        input_data (_type_): _description_
        num_inducing (int, optional): _description_. Defaults to 50.
        return_layers (bool, optional): _description_. Defaults to False.
        n_latent (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """

    num_data = input_data.shape[0]

    Z = input_data[np.random.choice(input_data.shape[0], size = num_inducing)]
    kernel1 = gpflow.kernels.SquaredExponential(lengthscales=[1]*input_data.shape[1])
    inducing_variable1 = gpflow.inducing_variables.InducingPoints(Z.copy())
    gp_layer1 = gpflux.layers.GPLayer(
        kernel1, inducing_variable1, num_data=num_data, num_latent_gps=n_latent, mean_function=gpflow.mean_functions.Zero()
    )

    kernel2 = gpflow.kernels.SquaredExponential(lengthscales=[1]*n_latent)
    inducing_variable2 = gpflow.inducing_variables.InducingPoints(np.random.rand(num_inducing,n_latent))
    gp_layer2 = gpflux.layers.GPLayer(
        kernel2,
        inducing_variable2,
        num_data=num_data,
        num_latent_gps=input_data.shape[1],
        mean_function=gpflow.mean_functions.Zero(),
    )

    likelihood_layer = gpflux.layers.LikelihoodLayer(gpflow.likelihoods.Gaussian(0.1))
    two_layer_dgp = gpflux.models.DeepGP([gp_layer1, gp_layer2], likelihood_layer)
    model = two_layer_dgp.as_training_model()
    model.compile(tf.optimizers.Adam(0.01))

    if return_layers:
        return model, gp_layer1, gp_layer2
    else:
        return model
    
def create_two_layer_GPAM_from_scratch(num_input, num_data = 1, Z = None, num_inducing = 50, return_layers = False, n_latent = 2):
    """_summary_

    Args:
        num_input (_type_): _description_
        num_data (int, optional): _description_. Defaults to 1.
        Z (_type_, optional): _description_. Defaults to None.
        num_inducing (int, optional): _description_. Defaults to 50.
        return_layers (bool, optional): _description_. Defaults to False.
        n_latent (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """

    if Z is not None:
        pass
    else:
        Z = np.random.rand(num_inducing, num_input)

    kernel1 = gpflow.kernels.SquaredExponential(lengthscales=[1]*num_input)
    inducing_variable1 = gpflow.inducing_variables.InducingPoints(Z.copy())
    gp_layer1 = gpflux.layers.GPLayer(
        kernel1, inducing_variable1, num_data=num_data, num_latent_gps=n_latent,mean_function=gpflow.mean_functions.Zero()
    )

    kernel2 = gpflow.kernels.SquaredExponential(lengthscales=[1]*n_latent)
    inducing_variable2 = gpflow.inducing_variables.InducingPoints(np.random.rand(num_inducing,n_latent))
    gp_layer2 = gpflux.layers.GPLayer(
        kernel2,
        inducing_variable2,
        num_data=num_data,
        num_latent_gps=num_input,
        mean_function=gpflow.mean_functions.Zero(),
    )

    likelihood_layer = gpflux.layers.LikelihoodLayer(gpflow.likelihoods.Gaussian(0.1))
    two_layer_dgp = gpflux.models.DeepGP([gp_layer1, gp_layer2], likelihood_layer)
    model = two_layer_dgp.as_training_model()
    model.compile(tf.optimizers.Adam(0.01))

    if return_layers:
        return model, gp_layer1, gp_layer2
    else:
        return model

def model_inference(data, encoder,batch_size=20000):
    """_summary_

    Args:
        data (_type_): _description_
        encoder (_type_): _description_
        batch_size (int, optional): _description_. Defaults to 20000.

    Returns:
        _type_: _description_
    """
    import tqdm
    max_iter = len(data)/batch_size
    means = []
    vars = []
    for i in tqdm.tqdm(range(int(max_iter)+1)):
        if max_iter - i < 0:
            res = encoder(data[batch_size*i:])
        else:
            res = encoder(data[batch_size*i:batch_size*(i+1)])

        means.append(res.mean())
        vars.append(res.variance())

    return np.concatenate(means, axis = 0), np.concatenate(vars, axis = 0)