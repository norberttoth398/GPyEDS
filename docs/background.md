# Background

## Dimensionality reduction for segmentation

Dimensionality reduction is the process of mapping data from a high dimensional space to a lower dimensional one with as little loss of information as possible. This process is ideal for working with high dimensional datasets that are inherently difficult to visualise and work with in general. In terms of segmentation, dimensionality reduction allows to circumvent the curse of dimensionality and to easily visualise the data during processing (eg. whilst clustering) allowing superior control over more conventional database-led approaches. It is also possible to add dimensionality reduction directly to conventional approaches.

## Autoencoders

Autoencoders are a general architecture of neural networks used for dimensinality reduction. The architecture is made of two distinct layers or functions:

	x' = f(g(x))

where f and g are decoder and encoder functions that are used to map the data, x, between high and low dimensional manifolds; note that f and g can be any neural network, generally multilayer perceptrons.

Autoencoders can be trained using reconstruction error, such as mean squared error. The weights of the model are tuned to minimise this reconstruction error. Therefore, the training goal is to retain as much information as possible whilst squeezing the data through a dimensionality bottleneck (the low dimensional manifold).

Autoencoders have previously been used for segmentation - see SIGMA (https://doi.org/10.1029/2022GC010530 and https://github.com/poyentung/sigma).

## Gaussian Processes (GPs)

Gaussian processes are used in the present software to replace the neural networks in the autoencoder models. They are a nonparametric supervised learning method used to solve regression and probabilistic classification problems. They benefit from the properties they inherit from the normal distribution. For more practical information see dedicated python-based GP software such as GPFlow, GPFlux and GPyTorch.
