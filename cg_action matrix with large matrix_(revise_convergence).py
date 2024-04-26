# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 10:22:03 2024

@author: thanh
"""

import numpy as np

# Define Action Matrix Function with a bigger matrix A
A = None  # Define A in the global scope

def action_matrix(x):
    global A  # Use the global A
    if A is not None:
        return A @ x
    else:
        A = np.random.randint(0, 5, size=(50, 50))  # Replace with specific matrix A
        A = (A + np.transpose(A))/2
    return A @ x

# Define Conjugate Gradient Function
def conjugate_gradient(b, x0=None, tol=1e-8, max_iter=None):
    if x0 is None:
        x0 = np.zeros_like(b)

    r = b - action_matrix(x0)
    p = r
    rs_old = np.dot(r, r)

    if max_iter is None:
        max_iter = len(b)

    for i in range(max_iter):
        Ap = action_matrix(p)
        alpha = rs_old / np.dot(p, Ap)
        x0 = x0 + alpha * p
        r = r - alpha * Ap
        rs_new = np.dot(r, r)

        print(f"Iteration {i + 1}: Residual Norm = {np.sqrt(rs_new)}")

        if np.sqrt(rs_new) < tol:
            print(f"Conjugate Gradient converged in {i + 1} iterations.")
            return x0

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    print("Conjugate Gradient did not converge within the specified number of iterations.")
    return x0

if __name__ == "__main__":
    # Given a larger matrix A (nxn) and vector b
    b = np.random.rand(50)  # Replace with specific n-dimensional vector b

    # Solve the system Ax = b using conjugate gradient with action_matrix
    max_iterations = 100  # Adjust the maximum number of iterations
    x_solution = conjugate_gradient(b, max_iter=max_iterations)

    print("Solution x:", x_solution)
