import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

# Historical data
data = {
    "Year": [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2022, 2023],
    "Winter": [144022, 157416, 155233, 161074, 168744, 186149, 186798, 189394, 190574, 188500, 262580],
    "Summer": [918533, 936116, 999365, 979187, 1050224, 1095950, 1136543, 1195544, 1305700, 1167000, 1670000],
    "Total": [1062555, 1093532, 1154598, 1140261, 1218968, 1282099, 1323341, 1384938, 1496274, 1355500, 1932580],
    "Average Temperature": [5.05, 4.73, 5.40, 5.66, 6.34, 6.85, 5.29, 6.11, 6.68, 5.99, 6.65],
    "Average Rainfall": [5.61, 5.13, 5.84, 5.02, 6.80, 5.18, 5.11, 3.85, 4.50, 6.14, 5.44]
}
df = pd.DataFrame(data)


def simulate_tourist_sde(mu=0.07, sigma=0.15, dt=1 / 252, T=5, n_paths=1000, V0=1932580, K=3000000):
    """
    Simulate tourist numbers using a stochastic differential equation (SDE)

    Parameters:
    mu: Growth rate
    sigma: Volatility
    dt: Time step
    T: Simulation duration in years
    n_paths: Number of simulation paths
    V0: Initial number of tourists
    K: Maximum carrying capacity
    """
    steps = int(T / dt)
    dW = norm.rvs(size=(steps, n_paths)) * np.sqrt(dt)
    V = np.zeros((steps + 1, n_paths))
    V[0] = V0

    # Euler-Maruyama method to solve the SDE
    for t in range(steps):
        V[t + 1] = V[t] + mu * V[t] * (1 - V[t] / K) * dt + sigma * V[t] * dW[t]

    time = np.linspace(0, T, steps + 1)
    mean_path = np.mean(V, axis=1)
    percentile_5 = np.percentile(V, 5, axis=1)
    percentile_95 = np.percentile(V, 95, axis=1)

    plt.figure(figsize=(12, 8))

    # Plot sample paths
    for i in range(10):
        plt.plot(time, V[:, i], alpha=0.2, color='gray', label='Sample Paths' if i == 0 else '')

    # Plot mean path and confidence intervals
    plt.plot(time, mean_path, 'b-', label='Mean Path', linewidth=2)
    plt.fill_between(time, percentile_5, percentile_95, color='b', alpha=0.1, label='95% CI')

    # Title and labels
    plt.title('Tourist Arrival Simulation (SDE Model)', fontsize=16, fontweight='bold')
    plt.xlabel('Time (years)', fontsize=14)
    plt.ylabel('Number of Tourists', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()  # Ensure everything fits nicely

    results = {
        'Mean': mean_path[-1],
        '5th Percentile': percentile_5[-1],
        '95th Percentile': percentile_95[-1]
    }

    # Save as PDF (vector graphic)
    plt.savefig('tourist_arrival_simulation.pdf', format='pdf')

    return plt.gcf(), results, V, time


def predict_future_visitors(years=5, V0=1932580, mu=0.07, sigma=0.15, K=3000000):
    """
    Predict the mean visitors for the next few years (starting from 2023)
    """
    future_years = np.arange(2024, 2024 + years)
    _, _, V, time = simulate_tourist_sde(mu=mu, sigma=sigma, T=years, V0=V0, K=K)

    # Calculate the mean number of visitors for each year
    mean_visitors_per_year = np.mean(V, axis=1)

    # Print the mean visitors for each future year
    for i, year in enumerate(future_years):
        print(f"Predicted Mean Visitors for {year}: {mean_visitors_per_year[i]:,.0f}")

    return future_years, mean_visitors_per_year


# Run the prediction and output results
future_years, predicted_mean_visitors = predict_future_visitors()
