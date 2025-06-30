import polars as pl
import numpy as np
from scipy.stats import norm
from scipy import optimize

def equations(x, S,greeks: dict):
    """
    Solve the GVV equations for given greeks and spot price S
    Args:
        x: tuple of (sigma, eta, rho)
        S: spot price
        greeks: dictionary containing greeks for options
    Returns a list of GVV equations evaluated at sigma, eta, rho.
    """
    sigma, eta, rho = x
    gvv = []

    # Loop through each option in greeks to calculate GVV equations
    for option in greeks.values():
        gamma = option['gamma']
        volga = option['volga']
        vanna = option['vanna']
        vol = option['vol']
        gvv.append(0.5 * gamma * S**2 * (sigma**2 - vol**2) + 0.5 * volga * vol**2 * eta**2 + vanna * S * vol * eta * sigma * rho)
    return gvv

def jacobian(x, S,greeks: dict):
    """
    Calculate the Jacobian matrix of the GVV equations
    Args:
        x: tuple of (sigma, eta, rho)
        S: spot price
        greeks: dictionary containing greeks for options
    Returns the Jacobian matrix evaluated at sigma, eta, rho.
    """
    sigma, eta, rho = x
    J = []

    for option in greeks.values():
        gamma = option['gamma']
        volga = option['volga']
        vanna = option['vanna']
        vol = option['vol']
        
        J.append([
            gamma * S**2 * sigma + vanna * S * vol * eta * rho,  # dF/dsigma
            volga * vol**2 * eta + vanna * S * vol * sigma * rho,  # dF/deta
            vanna * S * vol * eta * sigma  # dF/drho
        ])

    return J

def solve_gvv_least_square(greeks, S, initial_guess=(0.08, 0.2, -0.5)):
    """
    Solve the GVV system using least squares.
    Args:
        greeks: dictionary containing greeks for options
        S: spot price
        initial_guess: initial guess for (sigma, eta, rho)
    Returns sigma, eta, rho.
    """

    # Use scipy least_squares to solve the GVV equations
    solution = optimize.least_squares(equations, initial_guess, args=(S, greeks), jac=jacobian)
    
    if not solution.success:
        raise ValueError("Nonlinear solver did not converge")

    sigma, eta, rho = solution.x
    return sigma, eta, rho


def solve_gvv_linear(greeks, S):
    """
    Solve the GVV system using linear algebra approach.
    Args:
        greeks: dictionary containing greeks for options
        S: spot price
    Returns sigma, eta, rho.
    """
    # Extract coefficients from greeks dictionary
    coeffs = []
    vols = []
    
    # Loop through each option in greeks to build the coefficients
    for option in greeks.values():
        gamma = option['gamma']
        volga = option['volga']
        vanna = option['vanna']
        vol = option['vol']
        
        A_coeff = 0.5 * S**2 * gamma
        B_coeff = 0.5 * vol**2 * volga
        C_coeff = vanna * S * vol
        
        coeffs.append([A_coeff, B_coeff, C_coeff])
        vols.append(vol)
    
    # Build matrix A and vector b
    A = np.array(coeffs, dtype=float)
    b = np.array([coeffs[i][0] * vols[i]**2 for i in range(len(greeks))], dtype=float)
    
    # Solve system
    x, y, z = np.linalg.solve(A, b)
    
    # Extract params
    sigma = np.sqrt(x)
    eta = np.sqrt(y)
    rho = z / (sigma * eta)
    
    return sigma, eta, rho


def gvv_equation(vol_guess, S, K, T, r, q, sigma, eta, rho):
    """
    Calculate the GVV equation for a given volatility guess.
    Args:
        vol_guess: guessed volatility
        S: spot price
        K: strike price
        T: time to maturity
        r: risk-free interest rate
        q: dividend yield
        sigma: Implied underlying volatility
        eta: Implied volatility of the volatility
        rho: Imlied spot-volatility correlation
    Returns the value of the GVV equation.
    """
    # Calculate the greeks using the Black-Scholes 
    d1 = (np.log(S/K) + (r - q + 0.5 * vol_guess**2) * T) / (vol_guess * np.sqrt(T))
    d2 = d1 - vol_guess * np.sqrt(T)
    gamma = np.exp(-q * T) * norm.pdf(d1)/(S*vol_guess*np.sqrt(T))
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    volga = vega * (d1 * d2) / vol_guess
    vanna = (vega/S) * (1 - d1 / (vol_guess * np.sqrt(T))) 

    return 0.5 * gamma * S**2 * (sigma**2 - vol_guess**2) + 0.5 * volga * vol_guess**2 * eta**2 + vanna * S * vol_guess * eta * sigma * rho

def gvv_implied_vol(S, strikes, T, r, q,sigma, eta, rho, start=0.02, end=0.8):
    """
    Calculate the implied volatilities using the GVV equations.
    Args:
        S: spot price
        strikes: list of strike prices
        T: time to maturity
        r: risk-free interest rate
        q: dividend yield
        sigma: Implied underlying volatility
        eta: Implied volatility of the volatility
        rho: Imlied spot-volatility correlation
        start: starting point for the bisection method (default 0.4)
        end: ending point for the bisection method (default 0.8)
    Returns a list of implied volatilities for each strike.
    """
    implied_vols = []

    # Loop through each strike and calculate the implied vol
    for strike in strikes:
        # Use bisection method to find the implied volatility
        implied_vol = optimize.bisect(gvv_equation, start, end, args=(S, strike, T, r, q, sigma, eta, rho))
        implied_vols.append(implied_vol)
    return implied_vols