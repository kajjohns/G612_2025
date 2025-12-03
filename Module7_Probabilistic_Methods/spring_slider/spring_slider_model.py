"""
Spring-Slider Model with Rate-State Friction
Ported from MATLAB code for simulating slip evolution after sudden stress change
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


def rate_state(t, x, const):
    """
    Rate-state friction differential equations for spring-slider model
    
    Parameters:
    -----------
    t : float
        Time (years)
    x : array-like
        State vector [velocity, theta, tau, displacement]
        x[0] = v (slip velocity, m/yr)
        x[1] = theta (state variable, years)
        x[2] = tau (shear stress, MPa)
        x[3] = u (displacement, m)
    const : array-like
        Constants [d_c, A, B, sigma, k, v_inf, eta]
        
    Returns:
    --------
    dx/dt : array
        Time derivatives of state vector
    """
    # Unwrap constants
    d_c = const[0]      # critical displacement (m)
    A = const[1]        # rate-state parameter A
    B = const[2]        # rate-state parameter B
    sigma = const[3]    # normal stress (MPa)
    k = const[4]        # spring stiffness (MPa/m)
    v_inf = const[5]    # load-point velocity (m/yr)
    eta = const[6]      # radiation damping (MPa-yr/m)
    
    # Unwrap state variables
    v = x[0]            # velocity (m/yr)
    theta = x[1]        # state variable (years)
    tau = x[2]          # shear stress (MPa)
    
    # Calculate rates
    dtheta_dt = 1 - theta * v / d_c  # Aging Law
    # Alternative: Slip Law
    # dtheta_dt = -v * theta / d_c * np.log(v * theta / d_c)
    
    dtau_dt = k * (v_inf - v)
    
    dv_dt = (dtau_dt / sigma - B * dtheta_dt / theta) / (eta / sigma + A / v)
    
    du_dt = v  # displacement rate
    
    return np.array([dv_dt, dtheta_dt, dtau_dt, du_dt])


def forward_model(X, times):
    """
    Simulate fault slip evolution using spring-slider model with rate-state friction
    
    Parameters:
    -----------
    X : array-like
        Parameter vector [d_c, A, B, sigma, L, v_inf, deltau]
        d_c : critical displacement (m)
        A : rate-state parameter A
        B : rate-state parameter B
        sigma : normal stress (MPa)
        L : radius of slip patch (m)
        v_inf : load-point velocity (m/yr)
        deltau : instantaneous stress change (MPa)
    times : array-like
        Observation times (years)
        
    Returns:
    --------
    u : array
        Displacement at specified times (m)
    """
    # Unwrap parameters
    d_c = X[0]
    A = X[1]
    B = X[2]
    sigma = X[3]
    L = X[4]
    v_inf = X[5]
    deltau = X[6]
    
    # Calculate spring stiffness (L is radius of circular slip patch in m)
    k = 3e4 * 0.9 / L
    
    # Nominal friction coefficient
    mu_0 = 0.6
    
    # Seismic radiation damping term (MPa-yr/m)
    eta = 1.0e-7
    
    # INITIAL CONDITIONS
    # Steady state conditions before stress change
    v0 = v_inf  # initial slip rate (m/yr)
    tau0 = sigma * (mu_0 + (A - B) * np.log(v0))  # initial stress (MPa)
    theta0 = d_c / v0  # initial state variable (years)
    
    # Impose stress change
    tau = tau0 + deltau  # new initial stress immediately after stress change
    
    # Instantaneous velocity change (theta does not change instantaneously)
    v0 = v0 * np.exp(deltau / (A * sigma))
    
    # Initial state vector
    x0 = np.array([v0, theta0, tau0, 0.0])
    
    # Constants for ODE solver
    const = np.array([d_c, A, B, sigma, k, v_inf, eta])
    
    # Run ODE solver
    t0 = 0
    tf = times[-1]
    
    # Solve using scipy's solve_ivp (similar to MATLAB's ode23)
    sol = solve_ivp(
        fun=lambda t, x: rate_state(t, x, const),
        t_span=(t0, tf),
        y0=x0,
        method='RK23',  # Runge-Kutta 2(3) method, similar to ode23
        dense_output=True,
        rtol=1e-6,
        atol=1e-9
    )
    
    # Extract displacement
    u_solution = sol.sol(times)[3]  # 4th component is displacement
    
    return u_solution



