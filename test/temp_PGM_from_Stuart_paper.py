'''
build the PGM model like the Bayesian Relational Memory paper from Stuart Russell
'''
import pymc3 as pm
import numpy as np

# Generate some synthetic data
np.random.seed(123)
n_samples = 1000
a_true = 0.7
z_true = np.random.binomial(1, a_true, size=n_samples)
y_obs = np.random.binomial(1, z_true)

# Training data
z_train = z_true[:800]  # Use the first 800 samples for training

# Testing data
y_test = y_obs[800:]  # Use the remaining samples for testing

# Create the PyMC3 model
with pm.Model() as model:
    # Prior distribution for parameter a_prior
    a_prior = pm.Beta('a_prior', alpha=1, beta=1)

    # Latent variable z (training data)
    z = pm.Bernoulli('z', p=a_prior, shape=len(
        z_train), observed=z_train.reshape(-1))

    # Parameter a_obs
    a_obs = pm.Deterministic('a_obs', pm.math.switch(z, 1 - a_prior, a_prior))

    # Observation variable y (testing data)
    y = pm.Bernoulli('y', p=a_obs, shape=len(y_test), observed=y_test)

    # Perform maximum likelihood estimation for a_prior
    mle_estimate = pm.find_MAP()

    # Perform Bayesian inference
    trace = pm.sample(1000, tune=1000)

# Extract the posterior distribution of z
posterior_z = trace['z']

# Print the maximum likelihood estimate of a_prior
print("MLE estimate of a_prior:", mle_estimate['a_prior'])

# Print the posterior mean of z
print("Posterior mean of z:", posterior_z.mean(axis=0))
