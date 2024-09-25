# EDS2CHEM 

For detailed usage information please see example script.

## Markov Chain Monte Carlo (MCMC)

MCMC is an optimization method without direct optimization of a cost function. It instead starts by taking a random draw from a prior distribution. The parameter space is then repeatedly sampled to build a posterior distribution as according to a Markov Chain transition operator. This allows to construct an empirical distribution that converges to the exact posterior in the limit of infinite number of draws made. 

MCMC is advantageous in circumstances when posteriors are mathematically intractable. However results require careful examination as it is difficult to know when the posterior has converged without knowing the posterior itself. 

## Calibration Method

The EDS2CHEM calibration method makes use of the probabilistic nature of MCMC to build a distribution of possible profile locations on the sample and calibration curve. This helps to overcome the negative effects of noise and uncertainties in exact profile location. The resulting calibration propagates all uncertainties.
