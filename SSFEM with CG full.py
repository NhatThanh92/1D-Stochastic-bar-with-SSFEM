# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 18:35:42 2024

@author: thanh
"""

import numpy as np
from scipy.special import hermite
from scipy.integrate import quad
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

# Define the 1D stochastic bar problem
L = 1.0  # Length of the bar
E = 1.0  # Young's modulus
A = 1.0  # Cross-sectional area

# Number of elements for the spatial discretization
num_elements = 10


# Number of terms in the polynomial chaos expansion
num_terms = 5

# Define the multivariate Hermite polynomials
def hermite_polynomial(n, x):
    return hermite(n)(x * np.sqrt(2))

# Define the bilinear form for the stochastic Galerkin method
def bilinear_form(xi, eta):
    return E * A * (1 + xi) * (1 + eta)

# Perform quadrature using Gaussian quadrature
def perform_quadrature(func):
    quad_result, _ = quad(func, -1, 1)
    return quad_result

# Action matrix-vector product with Dirichlet boundary condition
def action_matrix(u):
    result = np.zeros_like(u)
    for i in range(num_elements):
        h = L / num_elements
        xi = (2 * i + 1) / (2 * num_elements) - 1 / np.sqrt(3)
        if i == 0:  # Dirichlet boundary condition at the first node
            continue
        for j in range(num_terms):
            eta = np.sqrt(2) * xi * np.sqrt(j + 1)
            weight = perform_quadrature(lambda x: hermite_polynomial(j, x)**2)
            result[j] += h * bilinear_form(xi, eta) * u[j] * weight
    return result

# Conjugate Gradient method
def conjugate_gradient(b, x0, tol=1e-6, max_iter=1000):
    x = x0.copy()
    r = b - action_matrix(x)
    p = r.copy()
    rs_old = np.dot(r, r)
    for i in range(max_iter):
        Ap = action_matrix(p)
        alpha = rs_old / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r, r)
        print(f"Iteration {i + 1}: Residual Norm = {np.sqrt(rs_new)}")
        if np.sqrt(rs_new) < tol:
            print(f"Conjugate Gradient converged in {i + 1} iterations.")
            return x
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    print("Conjugate Gradient did not converge within the specified number of iterations.")    
    return x

# Solve the 1D stochastic bar problem using SSFEM with PCE and conjugate gradient
def solve_stochastic_bar():
    # Initialize force vector
    f = np.zeros(num_terms)

    # External force assumed to be 1
    for j in range(num_terms):
        f[j] = L / num_elements

    # Modify force vector for tension force at the last node (right end)
    f[-1] = 1.0  # Tension force applied at the last node

    # Solve the linear system using conjugate gradient
    U = conjugate_gradient(f, np.zeros(num_terms))

    return U

if __name__ == "__main__":
    # Solve the stochastic bar problem
    U = solve_stochastic_bar()
    print("Displacements (U):", U)

    # Calculate KDE of displacements
    kde = gaussian_kde(U)

    # Plot KDE
    x = np.linspace(min(U), max(U), 100)
    plt.plot(x, kde(x), label='KDE of Displacements')
    plt.xlabel('Displacement')
    plt.ylabel('Density')
    plt.title('Kernel Density Estimation of Displacements')
    plt.legend()
    plt.grid(True)
    plt.show()
