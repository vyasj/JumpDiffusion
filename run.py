import matplotlib.pyplot as plt
import numpy as np
from math import erf

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
    
    plt.figure(1)
    plt.plot(paths.T.squeeze())
    plt.xlabel("$t$")
    plt.ylabel("Price")
    plt.title(f"$M$={M} Simulations of GBM with jumps of rate $\lambda$={lam}\n($\mu$={mu}, $\sigma$={sigma})")
    
    return paths

def Option_Price_BS(type: str, ul_path: list[float], K: float, T: int, r: float, vol: float) -> list[float]:
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
    Returns a path of option price with the same length as ul_path
    """
    path = []
    T = len(ul_path)
    if type.lower() == "call":
        for t, spot_price in enumerate(ul_path):
            d1 = (np.log(spot_price / K) + ((r + ((vol ** 2) / 2)) * (T-t))) / (vol * np.sqrt(T-t))
            d2 = d1 - (vol * np.sqrt(T-t))
            N = lambda x : (1 + erf(x / np.sqrt(2))) / 2
            path.append((spot_price * N(d1)) - (K * np.exp(-r * (T-t)) * N(d2)))
    return path

if __name__ == "__main__":
    S = 100
    mu = 0.1
    sigma = 0.25
    lam = 0.5
    T = 10
    dt = 0.1
    M = 1
    
    paths = Stock_GBM_Jumps(S, mu, sigma, lam, T, dt, M)
    
    K = 150
    r = 0.02
    
    for path in paths:
        plt.figure(2)
        plt.plot(Option_Price_BS("call", path, K, T//dt, r, sigma))
        plt.show()