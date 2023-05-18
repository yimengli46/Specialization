import numpy as np
from scipy.optimize import minimize

# Define the log-likelihood function for the Binomial distribution


def binomial_log_likelihood(p, k, n):
    eps = 1e-10  # Small constant to avoid zero in logarithm
    return -np.sum(k * np.log(p + eps) + (n - k) * np.log(1 - p + eps))


# Generate some synthetic data
np.random.seed(123)
n_trials = 1000
p_true = 0.7
k_successes = np.random.binomial(n=n_trials, p=p_true)

# Perform MLE to estimate p
result = minimize(binomial_log_likelihood, x0=0.5, args=(
    k_successes, n_trials), bounds=[(0, 1)])
mle_estimate = result.x[0]

# Print the Maximum Likelihood Estimate of p
print("MLE estimate of p:", mle_estimate)
