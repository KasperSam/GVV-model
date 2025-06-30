import polars as pl
import numpy as np
from scipy.stats import norm


# calcuate the greeks for the spot price
def bs_greeks(S, K, T, r, q, vol, option_type='call'):
    """
    Calculate the Black-Scholes option price and greeks for a European option.
    Args:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to expiration in years
        r (float): Risk-free interest rate (annualized)
        q (float): Dividend yield (annualized)
        vol (float): Volatility of the underlying asset (annualized)
        option_type (str): 'call' for call option, 'put' for put option
    Returns:
        dict: A dictionary containing the option price and greeks (delta, gamma, vega, theta, vanna, volga)
    """
    d1 = (np.log(S/K) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    cdf = norm.cdf

    # greeks
    if option_type == 'call':
        option_price = S * np.exp(-q * T) * cdf(d1) - K * np.exp(-r * T) * cdf(d2)
        delta = np.exp(-q * T) * norm.cdf(d1)
    else:
        option_price = K * np.exp(-r * T) * cdf(-d2) - S * np.exp(-q * T) * cdf(-d1)
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        
    gamma = np.exp(-q * T) * norm.pdf(d1)/(S*vol*np.sqrt(T))
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    volga = vega * (d1 * d2) / vol
    vanna = (vega/S) * (1 - d1 / (vol * np.sqrt(T))) 
    theta_base = -np.exp(-q * T) * S * norm.pdf(d1) * vol / (2 * np.sqrt(T))
    if option_type == 'call':
        theta = theta_base - (r * K * np.exp(-r * T) * norm.cdf(d2)) + (q * S * np.exp(-q * T) * norm.cdf(d1))
    else:
        theta = theta_base + (r * K * np.exp(-r * T) * norm.cdf(-d2)) - (q * S * np.exp(-q * T) * norm.cdf(-d1))


    return {'options_price': option_price,
            'implied_vol': vol,
            'delta': delta, 
            'vega': vega, 
            'gamma': gamma,
            'vanna': vanna, 
            'volga': volga,
            'theta': theta}



# calculate the greeks for the forward price
def bs_forward_greeks(F, K, T, r, vol, option_type='call'):
    """
    Calculate the Black-Scholes option price and greeks for a European option using forward price.
    Args:
        F (float): Forward price of the underlying asset
        K (float): Strike price
        T (float): Time to expiration in years
        r (float): Risk-free interest rate (annualized)
        vol (float): Volatility of the underlying asset (annualized)
        option_type (str): 'call' for call option, 'put' for put option
    Returns:
        dict: A dictionary containing the option price and greeks (delta, gamma, vega, theta, vanna, volga)
    """
    d1 = (np.log(F/K) + vol**2 * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    cdf = norm.cdf
    if option_type == 'call':
        option_price = np.exp(-r * T) * (F * cdf(d1) - K * cdf(d2))
        delta = np.exp(-r * T) * norm.cdf(d1)
    else:
        option_price = np.exp(-r * T) * (K * cdf(-d2) - F * cdf(-d1))
        delta = -np.exp(-r * T) * norm.cdf(-d1)

    vega = F * np.exp(-r * T) * norm.pdf(d1) * np.sqrt(T)
    gamma = np.exp(-r * T) * norm.pdf(d1)/(F*vol*np.sqrt(T))
    volga = vega * (d1 * d2) / vol
    vanna = (vega/F) * (1 - d1 / (vol * np.sqrt(T))) 
    theta_base = -np.exp(-r * T) * F * norm.pdf(d1) * vol / (2 * np.sqrt(T))

    if option_type == 'call':
        theta = theta_base - (r * K * np.exp(-r * T) * norm.cdf(d2)) + (r * F * np.exp(-r * T) * norm.cdf(d1))
    else:
        theta = theta_base + (r * K * np.exp(-r * T) * norm.cdf(-d2)) - (r * F * np.exp(-r * T) * norm.cdf(-d1))


    return {'options_price': option_price,
            'implied_vol': vol, 
            'delta': delta, 
            'vega': vega, 
            'gamma': gamma,
            'vanna': vanna, 
            'volga': volga,
            'theta': theta}

def calculate_greeks(data, S, T, r, q, option_type='call', forward_price=False):
    """
    Calculate the Black-Scholes option price and greeks for a list of options.
    Args:
        data (list of tuples): Each tuple contains (strike price, volatility)
        S (float): Current stock price
        T (float): Time to expiration in years
        r (float): Risk-free interest rate (annualized)
        q (float): Dividend yield (annualized)
        option_type (str): 'call' for call option, 'put' for put option
        forward_price (bool): If True, use forward price instead of spot price
    Returns:
        pl.DataFrame: A Polars DataFrame containing the option prices and greeks
    """
    results = {
        'options_prices': [],
        'implied_vols': [],
        'deltas': [],
        'vegas': [],
        'gammas': [],
        'vannas': [],
        'volgas': [],
        'thetas': []
    }

    strikes = []

    # Iterate through each option in the data
    for row in data:
        K = row[0] # strike price
        vol = row[1]
        strikes.append(K)
        
        # Get all Greeks at once
        if forward_price:
            greek_values = bs_forward_greeks(S, K, T, r, vol, option_type=option_type)
        else:
            greek_values = bs_greeks(S, K, T, r, q, vol, option_type=option_type)   
        
        # Append values to respective lists
        results['options_prices'].append(greek_values['options_price'])
        results['implied_vols'].append(greek_values['implied_vol'])
        results['deltas'].append(greek_values['delta'])
        results['vegas'].append(greek_values['vega'])
        results['gammas'].append(greek_values['gamma'])
        results['vannas'].append(greek_values['vanna'])
        results['volgas'].append(greek_values['volga'])
        results['thetas'].append(greek_values['theta'])

    df = pl.DataFrame(strikes, schema=['STRIKE'])

    # Add the results to the DataFrame
    if option_type == 'call':
        df = df.with_columns(pl.from_dict(results, ['C_PRICE', 'C_IMPLIED_VOL', 'C_DELTA', 'C_VEGA','C_GAMMA', 'C_VANNA', 'C_VOLGA', 'C_THETA']))
    else:
        df = df.with_columns(pl.from_dict(results, ['P_PRICE', 'P_IMPLIED_VOL', 'P_DELTA', 'P_VEGA', 'P_GAMMA', 'P_VANNA', 'P_VOLGA', 'P_THETA']))
    
    return df


# Newton-Raphson method
def find_implied_volatility(S, K, T, r, market_price, q=0.0, option_type='call', forward_price=False, initial_guess=0.2, max_iterations=100, precision=0.00001):
    """
    Calculate the implied volatility using the Newton-Raphson method.
    Args:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to expiration in years
        r (float): Risk-free interest rate (annualized)
        market_price (float): Market price of the option
        q (float): Dividend yield (annualized)
        option_type (str): 'call' for call option, 'put' for put option
        forward_price (bool): If True, use forward price instead of spot price
        initial_guess (float): Initial guess for volatility
        max_iterations (int): Maximum number of iterations to prevent infinite loops
        precision (float): Desired precision for convergence
    Returns:
        dict: A dictionary containing the implied volatility and the option price at each iteration
    """
    sigma = initial_guess
    prev_diff = float('inf')  # start with an infinite difference
    
    implied_vols = {}
    iteration = 0
    
    while iteration < max_iterations:
        # Use forward price calculations
        if forward_price:
            d1 = (np.log(S/K) + sigma**2 * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            cdf = norm.cdf
            if option_type == 'call':
                option_price = np.exp(-r * T) * (S * cdf(d1) - K * cdf(d2))
            else:
                option_price = np.exp(-r * T) * (K * cdf(-d2) - S * cdf(-d1))
            vega = S * np.exp(-r * T) * norm.pdf(d1) * np.sqrt(T)

        # Use spot price calculations
        else:
            d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            cdf = norm.cdf
            if option_type == 'call':
                option_price = S * np.exp(-q * T) * cdf(d1) - K * np.exp(-r * T) * cdf(d2)
            else:
                option_price = K * np.exp(-r * T) * cdf(-d2) - S * np.exp(-q * T) * cdf(-d1)
            vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

            
        # Calculate absolute difference
        current_diff = abs(option_price - market_price)
        
        # Store results
        implied_vols[iteration] = {
            'sigma': sigma,
            'price': option_price,
            'diff': round(current_diff, 5)
        }
        
        
        # Check if reached desired precision
        if current_diff < precision:
            # print(f"Converged with precision {precision} after {iteration} iterations")
            break
            
        # Check if getting worse
        if current_diff > prev_diff:
            print(f"Difference increasing, stopping at iteration {iteration}")
            # Revert to previous sigma since it was better
            sigma = implied_vols[iteration-1]['sigma']
            option_price = implied_vols[iteration-1]['price']
            break
            
        # Update for next iteration
        prev_diff = current_diff
        
        # Avoid division by zero or very small vega
        if abs(vega) < 1e-10:
            print("Vega too small, stopping")
            break
            
        # Update sigma using Newton-Raphson
        sigma = sigma - (option_price - market_price) / vega
        
        # Ensure sigma stays positive
        sigma = max(0.001, sigma)
        
        iteration += 1
    
    if iteration == max_iterations:
        print(f"Maximum iterations ({max_iterations}) reached without convergence")

    return implied_vols

def find_implied_volatilities(data, S, T, r, q=0.0, option_type='call', forward_price=False, initial_guess=0.2, max_iterations=100, precision=0.0001):
    """
    Calculate the implied volatilities for a list of options using the Newton-Raphson method.
    Args:
        data (list of tuples): Each tuple contains (strike price, market price)
        S (float): Current stock price
        T (float): Time to expiration in years
        r (float): Risk-free interest rate (annualized)
        q (float): Dividend yield (annualized)
        option_type (str): 'call' for call option, 'put' for put option
        forward_price (bool): If True, use forward price instead of spot price
        initial_guess (float): Initial guess for volatility
        max_iterations (int): Maximum number of iterations to prevent infinite loops
        precision (float): Desired precision for convergence
    Returns:
        pl.DataFrame: A Polars DataFrame containing the implied volatilities and option prices for each strike
    """
    results = {
        'strikes': [],
        'implied_vols': [],
        'prices': [],
        'diffs': []
    }

    # Iterate through each option in the data
    for row in data:
        K = row[0] # strike price
        market_price = row[1]
        
        implied_vols = find_implied_volatility(S, K, T, r, market_price, q, option_type, forward_price,
                                                initial_guess, max_iterations, precision)
        
        # Get the last iteration's results
        last_iteration = max(implied_vols.keys())
        results['strikes'].append(K)
        results['implied_vols'].append(implied_vols[last_iteration]['sigma'])
        results['prices'].append(implied_vols[last_iteration]['price'])
        results['diffs'].append(implied_vols[last_iteration]['diff'])

    df = pl.DataFrame({
        'STRIKE': results['strikes'],
        'IMPLIED_VOL': results['implied_vols'],
        'PRICE': results['prices'],
        'DIFF': results['diffs']
    })

    return df