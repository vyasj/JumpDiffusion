import matplotlib.pyplot as plt
import numpy as np

def GBM_Jumps(S0: int, mu: float, sigma: float, lam: float, T: int, dt: float, M: int) -> None:
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
    """
    paths = S0 * np.ones(shape=(M, int(T/dt)))
    for col in range(1, int(T/dt)):
        W_t = np.random.normal(0, np.sqrt(dt), size=(M, ))
        J_t = np.random.poisson(lam, size=(M, ))
        paths[:,col] = paths[:,col-1] * np.exp((mu - ((sigma**2)/2)) * dt + (sigma * W_t)) + J_t
    
    plt.plot(paths.T.squeeze())
    plt.xlabel("$t$")
    plt.ylabel("Price")
    plt.title(f"$M$={M} Simulations of GBM with jumps of rate $\lambda$={lam}\n($\mu$={mu}, $\sigma$={sigma})")
    plt.show()
    return

if __name__ == "__main__":
    S = 100
    mu = 0.1
    sigma = 0.25
    lam = 0.5
    T = 10
    dt = 0.1
    M = 5
    
    GBM_Jumps(S, mu, sigma, lam, T, dt, M)