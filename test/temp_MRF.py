import numpy as np
from scipy.optimize import minimize

# Define the MRF model parameters
theta = 0.5  # Unary potential parameter
beta = 1.0   # Pairwise potential parameter

# Generate synthetic noisy image
image = np.array([[0, 1, 1, 0],
                  [0, 1, 0, 0],
                  [1, 0, 1, 1]])

# Add random noise to the image
noisy_image = image + np.random.randint(low=-1, high=2, size=image.shape)

# Define the energy function for the MRF


def energy(x):
    unary_potential = -theta * np.sum(x * noisy_image)
    pairwise_potential = -beta * np.sum(x[:-1, :] * x[1:, :]) - beta * np.sum(x[:, :-1] * x[:, 1:])
    return unary_potential + pairwise_potential

# Define the hinge loss function


def hinge_loss(x):
    return np.maximum(0, 1 - x)

# Define the objective function to be minimized


def objective(x):
    return np.mean(hinge_loss(energy(x)))

# Define the constraint function for binary variables


def constraint(x):
    return x**2 - x


# Initialize random binary image
x0 = np.random.randint(0, 2, size=image.shape)

# Perform optimization to learn the MRF parameters
result = minimize(objective, x0, method='SLSQP', constraints={'type': 'eq', 'fun': constraint})
learned_image = result.x.reshape(image.shape)

# Print the original, noisy, and denoised images
print("Original Image:\n", image)
print("Noisy Image:\n", noisy_image)
print("Denoised Image:\n", learned_image.round())
