#!/usr/bin/env python3
"""
Underdetermined Least Squares Problem: Cubic Polynomial Example

This script demonstrates an underdetermined least squares problem where we attempt 
to fit a cubic polynomial (4 parameters) using fewer than 4 data points, creating
an underdetermined system with infinitely many solutions.

Author: Created for G612 Course Module 2
Date: September 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq, null_space
import warnings

def create_underdetermined_system():
    """
    Create an underdetermined system with 3 data points and 4 polynomial parameters.
    This creates a system where we have more unknowns than equations.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # True coefficients for cubic polynomial: y = a0 + a1*x + a2*x^2 + a3*x^3
    true_coeffs = np.array([1.0, -2.0, 0.5, 0.7])
    
    # Create only 3 data points (underdetermined: 3 equations, 4 unknowns)
    x_data = np.array([-1.0, 0.0, 1.0])
    
    # Generate true y values without noise first
    y_true = (true_coeffs[0] + 
              true_coeffs[1] * x_data + 
              true_coeffs[2] * x_data**2 + 
              true_coeffs[3] * x_data**3)
    
    # Add small amount of noise
    noise = np.random.normal(0, 0.1, len(x_data))
    y_data = y_true + noise
    
    # Build the design matrix G (3x4 matrix - underdetermined)
    G = np.vstack([np.ones_like(x_data), x_data, x_data**2, x_data**3]).T
    
    return x_data, y_data, G, true_coeffs, y_true

def solve_underdetermined_system(G, y_data, method='lstsq'):
    """
    Solve the underdetermined system using different approaches.
    
    Parameters:
    -----------
    G : ndarray
        Design matrix (3x4)
    y_data : ndarray
        Observed data (3x1)
    method : str
        Solution method ('lstsq', 'pinv', 'normal_eq')
    
    Returns:
    --------
    m_solution : ndarray
        Model parameters solution
    """
    
    if method == 'lstsq':
        # Use scipy's least squares (handles underdetermined case)
        m_solution, residuals, rank, s = lstsq(G, y_data, rcond=None)
        return m_solution, rank, s
        
    elif method == 'pinv':
        # Use Moore-Penrose pseudoinverse
        G_pinv = np.linalg.pinv(G)
        m_solution = G_pinv @ y_data
        rank = np.linalg.matrix_rank(G)
        s = np.linalg.svd(G, compute_uv=False)
        return m_solution, rank, s
        
    elif method == 'normal_eq':
        # Try normal equations (will fail for underdetermined case)
        try:
            GTG = G.T @ G
            GTd = G.T @ y_data
            # This will be singular for underdetermined systems
            m_solution = np.linalg.solve(GTG, GTd)
            rank = np.linalg.matrix_rank(GTG)
            s = np.linalg.svd(GTG, compute_uv=False)
            return m_solution, rank, s
        except np.linalg.LinAlgError:
            print("Normal equations failed (singular matrix) - expected for underdetermined system!")
            return None, None, None

def explore_solution_space(G, y_data):
    """
    Explore the null space of the underdetermined system to show 
    infinitely many solutions.
    """
    # Get the minimum norm solution
    m_min_norm, rank, s = solve_underdetermined_system(G, y_data, 'pinv')
    
    # Find the null space of G
    null_vectors = null_space(G)
    
    print(f"System properties:")
    print(f"  Design matrix shape: {G.shape}")
    print(f"  Matrix rank: {rank}")
    print(f"  Null space dimension: {null_vectors.shape[1]}")
    print(f"  Singular values: {s}")
    
    # Generate alternative solutions by adding null space components
    alternative_solutions = []
    alphas = [-2, -1, 0, 1, 2]  # Different scaling factors
    
    for alpha in alphas:
        if null_vectors.shape[1] > 0:
            m_alt = m_min_norm + alpha * null_vectors[:, 0]  # Use first null vector
            alternative_solutions.append(m_alt)
        else:
            alternative_solutions.append(m_min_norm)
    
    return m_min_norm, alternative_solutions, alphas, null_vectors

def plot_underdetermined_results(x_data, y_data, true_coeffs, m_min_norm, 
                                alternative_solutions, alphas):
    """
    Plot the results showing multiple solutions to the underdetermined problem.
    """
    # Create fine grid for plotting
    x_plot = np.linspace(-2, 2, 200)
    
    # True model
    y_true_plot = (true_coeffs[0] + 
                   true_coeffs[1] * x_plot + 
                   true_coeffs[2] * x_plot**2 + 
                   true_coeffs[3] * x_plot**3)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Multiple solutions
    ax1.scatter(x_data, y_data, color='red', s=100, zorder=5, 
                label='Observed data (3 points)', edgecolors='black')
    ax1.plot(x_plot, y_true_plot, 'b-', linewidth=2, label='True model')
    
    # Plot minimum norm solution
    y_min_norm = (m_min_norm[0] + 
                  m_min_norm[1] * x_plot + 
                  m_min_norm[2] * x_plot**2 + 
                  m_min_norm[3] * x_plot**3)
    ax1.plot(x_plot, y_min_norm, 'g--', linewidth=2, 
             label='Minimum norm solution')
    
    # Plot alternative solutions
    colors = ['purple', 'orange', 'brown', 'pink', 'gray']
    for i, (m_alt, alpha, color) in enumerate(zip(alternative_solutions, alphas, colors)):
        if alpha != 0:  # Skip the minimum norm solution (alpha=0)
            y_alt = (m_alt[0] + 
                     m_alt[1] * x_plot + 
                     m_alt[2] * x_plot**2 + 
                     m_alt[3] * x_plot**3)
            ax1.plot(x_plot, y_alt, '--', color=color, alpha=0.7,
                     label=f'Alternative solution (α={alpha})')
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Underdetermined System: Multiple Solutions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-2, 2)
    
    # Plot 2: Residuals for each solution
    residuals = []
    for m_alt in alternative_solutions:
        G_plot = np.vstack([np.ones_like(x_data), x_data, x_data**2, x_data**3]).T
        y_pred = G_plot @ m_alt
        residual = np.linalg.norm(y_data - y_pred)
        residuals.append(residual)
    
    ax2.bar(range(len(alphas)), residuals, color=colors)
    ax2.set_xlabel('Solution Index')
    ax2.set_ylabel('Residual Norm ||d - Gm||')
    ax2.set_title('Residuals for Different Solutions')
    ax2.set_xticks(range(len(alphas)))
    ax2.set_xticklabels([f'α={α}' for α in alphas])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def demonstrate_rank_deficiency():
    """
    Demonstrate what happens when we try different numbers of data points.
    """
    print("\n" + "="*60)
    print("RANK DEFICIENCY ANALYSIS")
    print("="*60)
    
    # True coefficients
    true_coeffs = np.array([1.0, -2.0, 0.5, 0.7])
    
    for n_points in [2, 3, 4, 5]:
        print(f"\nSystem with {n_points} data points:")
        
        # Create data points
        x = np.linspace(-1, 1, n_points)
        y_true = (true_coeffs[0] + 
                  true_coeffs[1] * x + 
                  true_coeffs[2] * x**2 + 
                  true_coeffs[3] * x**3)
        y = y_true + np.random.normal(0, 0.05, n_points)
        
        # Build design matrix
        G = np.vstack([np.ones_like(x), x, x**2, x**3]).T
        
        # Analyze the system
        rank = np.linalg.matrix_rank(G)
        condition_number = np.linalg.cond(G)
        
        print(f"  Design matrix shape: {G.shape}")
        print(f"  Matrix rank: {rank}")
        print(f"  Condition number: {condition_number:.2e}")
        
        if n_points < 4:
            print(f"  System type: UNDERDETERMINED ({n_points} equations, 4 unknowns)")
            print(f"  Null space dimension: {4 - rank}")
        elif n_points == 4:
            if rank == 4:
                print(f"  System type: EXACTLY DETERMINED (unique solution)")
            else:
                print(f"  System type: RANK DEFICIENT (infinite solutions)")
        else:
            print(f"  System type: OVERDETERMINED ({n_points} equations, 4 unknowns)")
            print(f"  Least squares solution exists")

def main():
    """
    Main function to demonstrate underdetermined least squares problem.
    """
    print("="*60)
    print("UNDERDETERMINED LEAST SQUARES: CUBIC POLYNOMIAL EXAMPLE")
    print("="*60)
    
    # Create underdetermined system
    x_data, y_data, G, true_coeffs, y_true = create_underdetermined_system()
    
    print(f"Problem setup:")
    print(f"  True coefficients: {true_coeffs}")
    print(f"  Data points: {len(x_data)} (x: {x_data})")
    print(f"  Observed y values: {y_data}")
    print(f"  Design matrix G shape: {G.shape}")
    print(f"  Problem type: {G.shape[0]} equations, {G.shape[1]} unknowns")
    
    print(f"\nDesign matrix G:")
    print(G)
    
    # Explore solution space
    m_min_norm, alternative_solutions, alphas, null_vectors = explore_solution_space(G, y_data)
    
    print(f"\nMinimum norm solution: {m_min_norm}")
    print(f"True coefficients:     {true_coeffs}")
    print(f"Error in solution:     {np.linalg.norm(m_min_norm - true_coeffs):.4f}")
    
    if null_vectors.shape[1] > 0:
        print(f"\nNull space vectors:")
        for i in range(null_vectors.shape[1]):
            print(f"  Null vector {i+1}: {null_vectors[:, i]}")
        
        print(f"\nAlternative solutions (m = m_min_norm + α * null_vector):")
        for alpha, m_alt in zip(alphas, alternative_solutions):
            residual = np.linalg.norm(y_data - G @ m_alt)
            print(f"  α={alpha:2d}: m={m_alt}, residual={residual:.6f}")
    
    # Try different solution methods
    print(f"\n" + "-"*40)
    print("COMPARISON OF SOLUTION METHODS")
    print("-"*40)
    
    # Method 1: scipy lstsq
    m_lstsq, rank_lstsq, s_lstsq = solve_underdetermined_system(G, y_data, 'lstsq')
    print(f"scipy.linalg.lstsq solution: {m_lstsq}")
    
    # Method 2: pseudoinverse
    m_pinv, rank_pinv, s_pinv = solve_underdetermined_system(G, y_data, 'pinv')
    print(f"Moore-Penrose pseudoinverse: {m_pinv}")
    
    # Method 3: normal equations (will fail)
    print(f"Normal equations: ", end="")
    m_normal, rank_normal, s_normal = solve_underdetermined_system(G, y_data, 'normal_eq')
    if m_normal is not None:
        print(f"{m_normal}")
    
    # Verify solutions give same residual
    print(f"\nResidual verification:")
    print(f"  lstsq residual:      {np.linalg.norm(y_data - G @ m_lstsq):.10f}")
    print(f"  pseudoinverse residual: {np.linalg.norm(y_data - G @ m_pinv):.10f}")
    
    # Plot results
    plot_underdetermined_results(x_data, y_data, true_coeffs, m_min_norm, 
                                alternative_solutions, alphas)
    
    # Demonstrate rank deficiency for different numbers of points
    demonstrate_rank_deficiency()
    
    print(f"\n" + "="*60)
    print("KEY INSIGHTS:")
    print("="*60)
    print("1. Underdetermined systems have infinitely many solutions")
    print("2. The pseudoinverse gives the minimum norm solution")
    print("3. Any solution can be written as: m = m_min_norm + α*null_vector")
    print("4. All solutions fit the data equally well (same residual)")
    print("5. Normal equations fail due to singular G^T*G matrix")
    print("6. Additional constraints or regularization needed for unique solution")

if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    main()
