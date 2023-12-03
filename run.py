import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def Stock_GBM_Jumps(S0: int, mu: float, sigma: float, lam: float, T: int, dt: float, M: int) -> list[list[float]]:
    """
    Performs M-Simulations of GBM with Jumps
    ----------------------------------------
    S0 : initial stock price
    mu : mean of stock price distribution
    sigma : variance of stock price distribution
    lam : arrival rate of Poisson distribution for jumps
    T : total time
    dt : step size
    M : number of simulations
    ----------------------------------------
    Returns a matrix of M paths with int(T/dt) steps
    """
    paths = S0 * np.ones(shape=(M, int(T/dt)))
    for col in range(1, int(T/dt)):
        W_t = np.random.normal(0, np.sqrt(dt), size=(M, ))
        J_t = S0 * np.random.poisson(lam, size=(M, ))
        paths[:,col] = paths[:,col-1] * np.exp((mu - ((sigma**2)/2)) * dt + (sigma * W_t)) + J_t
    
    return np.array(paths)


def Option_Price_BS(type: str, ul_paths: list[list[float]], K: float, T: int, r: float, vol: float) -> list[list[float]]:
    """
    Calculates the price of an option using the Black-Scholes formula
    -----------------------------------------------------------------
    type : type of option, either "call" or "put"
    ul_path : price path of the underlying stock
    K : strike price
    T : time to maturity
    r : risk free rate
    vol : volatility of underlying asset
    -----------------------------------------------------------------
    Returns an option price path for each underlying asset path in ul_paths
    """
    N = norm.cdf
    option_paths = []
    for ul_path in ul_paths:
        d1 = (np.log(ul_path / K) + (r + (vol**2) / 2) * T) / (vol * np.sqrt(T))
        d2 = d1 - (vol * np.sqrt(T))
        
        if type == "call":
            option_paths.append((ul_path * N(d1)) - (K * np.exp(-r * T) * N(d2)))
        elif type == "put":
            option_paths.append((K * np.exp(-r * T) * N(d2)) - (ul_path * N(-d1)))
        else:
            print("Invalid option type, must be either 'call' or 'put'.")
            exit()
        
    return np.array(option_paths)


if __name__ == "__main__":
    S = 100
    mu = 0.1
    sigma = 0.25
    lam = 0.5
    T = 10
    dt = 0.1
    M = 1
    
    paths = Stock_GBM_Jumps(S, mu, sigma, lam, T, dt, M)
    
    plt.figure(1)
    plt.plot(paths.T.squeeze())
    plt.xlabel("$t$")
    plt.ylabel("Price")
    plt.title(f"$M$={M} Simulations of GBM with jumps of rate $\lambda$={lam}\n($\mu$={mu}, $\sigma$={sigma})")
    
    K = 150
    r = 0.02
    type = "call"
    
    opt_paths = Option_Price_BS(type, paths, K, T, r, sigma)
    
    plt.figure(2)
    plt.plot(opt_paths.T.squeeze())
    plt.xlabel("$t$")
    plt.ylabel("Price")
    plt.title(f"$M$={M} {type} options of underlying asset paths from Figure 1")
    
    plt.show()