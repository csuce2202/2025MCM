import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
import pandas as pd
from sklearn.metrics import r2_score
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

# Historical data
data = {
    "Year": [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2022, 2023],
    "Winter": [182000,214000,275000,320000,410000,550000,690000,720000,620000,530000,680000],
    "Summer": [388000,456000,535000,680000,880000,1240000,1540000,1620000,1390000,1170000,1540000],
    "Total": [570000,670000,810000,1000000,1290000,1790000,2230000,2340000,2010000,1700000,2220000],
    "Average Temperature": [0.05344979,0.3218217,-0.056156296,1.1464639,-0.37374434,1.026877,0.860888,0.3407371,0.25315195,-0.09324447,-0.2669504],
    "Average Rainfall": [3.145205479,3.235616438,3,3.780821918,3.284931507,3.438356164,3.315068493,3.41369863,3.208219178,3.34,3.586]
}
df = pd.DataFrame(data)


def simulate_tourist_sde(mu=0.272, sigma=0.251, dt=1 / 252, T=12, n_paths=1000, V0=570000, K=3000000):
    """
    Simulate tourist numbers using a stochastic differential equation (SDE)
    Starting from 2011 (V0=1062555) and simulating until 2023
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

    return V, time, mean_path, percentile_5, percentile_95


def plot_historical_simulation(V, time, mean_path, percentile_5, percentile_95, df):
    plt.figure(figsize=(12, 8))

    # Plot sample paths
    for i in range(10):
        plt.plot(time + 2011, V[:, i], alpha=0.2, color='gray', label='Sample Paths' if i == 0 else '')

    # Plot mean path and confidence intervals
    plt.plot(time + 2011, mean_path, 'b-', label='Simulated Mean Path', linewidth=2)
    plt.fill_between(time + 2011, percentile_5, percentile_95, color='b', alpha=0.1, label='95% CI')

    # Plot actual historical data
    plt.plot(df['Year'], df['Total'], 'r-o', label='Actual Historical Data', linewidth=2)

    # Title and labels
    plt.title('Tourist Arrival Simulation vs Historical Data (2011-2023)', fontsize=16, fontweight='bold')
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Number of Tourists', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Format y-axis with comma separator
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    plt.tight_layout()
    return plt.gcf()


def simulate_future(mu=0.07, sigma=0.15, dt=1 / 252, T=5, n_paths=1000, V0=1932580, K=3000000):
    """
    Simulate future tourist numbers starting from 2023
    """
    steps = int(T / dt)
    dW = norm.rvs(size=(steps, n_paths)) * np.sqrt(dt)
    V = np.zeros((steps + 1, n_paths))
    V[0] = V0

    # Euler-Maruyama method
    for t in range(steps):
        V[t + 1] = V[t] + mu * V[t] * (1 - V[t] / K) * dt + sigma * V[t] * dW[t]

    time = np.linspace(0, T, steps + 1)
    mean_path = np.mean(V, axis=1)
    percentile_5 = np.percentile(V, 5, axis=1)
    percentile_95 = np.percentile(V, 95, axis=1)

    plt.figure(figsize=(12, 8))

    # Plot sample paths
    for i in range(10):
        plt.plot(time + 2023, V[:, i], alpha=0.2, color='gray', label='Sample Paths' if i == 0 else '')

    # Plot mean path and confidence intervals
    plt.plot(time + 2023, mean_path, 'g-', label='Forecasted Mean Path', linewidth=2)
    plt.fill_between(time + 2023, percentile_5, percentile_95, color='g', alpha=0.1, label='95% CI')

    # Plot the 2023 starting point
    plt.plot(2023, V0, 'ro', label='2023 Starting Point', markersize=10)

    # Title and labels
    plt.title('Tourist Arrival Forecast (2023-2028)', fontsize=16, fontweight='bold')
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Number of Tourists', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Format y-axis with comma separator
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    plt.tight_layout()

    # Prepare forecast data
    forecast_years = np.arange(2023, 2029)
    forecast_values = np.interp(forecast_years, time + 2023, mean_path)

    return plt.gcf(), forecast_years, forecast_values


def calculate_error_metrics(df, simulated_years, mean_path):
    """
    Calculate error metrics between simulated and actual data
    """
    actual_data = df.set_index('Year')['Total']

    # Get yearly averages from simulation
    yearly_means = pd.Series(index=np.unique(simulated_years))
    for year in yearly_means.index:
        mask = simulated_years == year
        yearly_means[year] = np.mean(mean_path[mask])

    # Calculate metrics for matching years
    matching_years = yearly_means.index.intersection(actual_data.index)
    actual_values = actual_data[matching_years]
    predicted_values = yearly_means[matching_years]

    # Calculate MAPE, RMSE, and R²
    mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
    rmse = np.sqrt(np.mean((actual_values - predicted_values) ** 2))
    r2 = r2_score(actual_values, predicted_values)

    return {
        'MAPE': mape,
        'RMSE': rmse,
        'R2': r2,
        'Yearly_Comparison': pd.DataFrame({
            'Actual': actual_values,
            'Predicted': predicted_values,
            'Error_Percent': ((predicted_values - actual_values) / actual_values) * 100
        })
    }


def create_3d_sensitivity_analysis(df, mu_range, sigma_range, T=12, base_V0=1062555, K=3000000):
    """
    Create publication-quality 3D visualization of sensitivity analysis
    with fixes for overlapping labels and enhanced visual elements
    """
    # Set publication-quality style
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 300
    })

    # Create time points
    time_points = np.linspace(0, T, 13)  # 13 points for 12 years

    # Initialize arrays to store results
    mu_results = []
    sigma_results = []

    # Perform analysis for mu
    for mu in mu_range:
        tourist_numbers = []
        V, time, mean_path, _, _ = simulate_tourist_sde(
            mu=mu, sigma=0.15,  # Fix sigma to analyze mu's effect
            V0=base_V0, K=K, T=T
        )
        tourist_numbers = mean_path[::int(len(mean_path) / 13)][:13]  # Ensure exactly 13 points
        mu_results.append(tourist_numbers)

    # Perform analysis for sigma
    for sigma in sigma_range:
        tourist_numbers = []
        V, time, mean_path, _, _ = simulate_tourist_sde(
            mu=0.07, sigma=sigma,  # Fix mu to analyze sigma's effect
            V0=base_V0, K=K, T=T
        )
        tourist_numbers = mean_path[::int(len(mean_path) / 13)][:13]  # Ensure exactly 13 points
        sigma_results.append(tourist_numbers)

    # Create figure with enhanced layout
    fig = plt.figure(figsize=(20, 8))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.3)

    # Plot for mu sensitivity
    ax1 = fig.add_subplot(gs[0], projection='3d')
    ax1.set_facecolor('white')
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Plot vertical planes for mu with enhanced aesthetics
    for i, mu in enumerate(mu_range):
        z_values = np.array(mu_results[i])
        x_values = time_points
        y_values = np.full_like(x_values, mu)

        # Create vertical plane
        X, Y = np.meshgrid([x_values[0], x_values[-1]], [mu, mu])
        Z = np.array([np.zeros_like([x_values[0], x_values[-1]]),
                     [z_values[0], z_values[-1]]])

        # Dynamic color with enhanced transparency
        color = plt.cm.viridis(i / len(mu_range))

        # Plot vertical plane and line with improved styling
        ax1.plot_surface(X, Y, Z, alpha=0.3, color=color, edgecolor='none')
        ax1.plot(x_values, y_values, z_values,
                linestyle='--',
                marker='o',
                markersize=4,
                color=color,
                linewidth=1.5)

    # Enhanced colorbar with adjusted position
    sm1 = plt.cm.ScalarMappable(cmap='viridis',
                               norm=plt.Normalize(vmin=min(mu_range), vmax=max(mu_range)))
    sm1.set_array([])
    cbar1 = plt.colorbar(sm1, ax=ax1, shrink=0.8, aspect=20, pad=0.1)
    cbar1.set_label('Growth Rate (μ)', fontsize=12, labelpad=10)

    # Enhanced labels with adjusted positions
    ax1.set_xlabel('Time (Years from 2011)', labelpad=15)
    ax1.set_ylabel('μ Value', labelpad=15)
    ax1.set_zlabel('Tourist Numbers',labelpad=5)
    ax1.set_title('(a) Sensitivity Analysis of Growth Rate (μ)', pad=20)

    # Optimized view angle
    ax1.view_init(elev=25, azim=-60)

    # Plot for sigma sensitivity
    ax2 = fig.add_subplot(gs[1], projection='3d')
    ax2.set_facecolor('white')
    ax2.grid(True, linestyle='--', alpha=0.3)

    # Plot vertical planes for sigma with enhanced aesthetics
    for i, sigma in enumerate(sigma_range):
        z_values = np.array(sigma_results[i])
        x_values = time_points
        y_values = np.full_like(x_values, sigma)

        # Create vertical plane
        X, Y = np.meshgrid([x_values[0], x_values[-1]], [sigma, sigma])
        Z = np.array([np.zeros_like([x_values[0], x_values[-1]]),
                     [z_values[0], z_values[-1]]])

        # Dynamic color with enhanced transparency
        color = plt.cm.viridis(i / len(sigma_range))

        # Plot vertical plane and line with improved styling
        ax2.plot_surface(X, Y, Z, alpha=0.3, color=color, edgecolor='none')
        ax2.plot(x_values, y_values, z_values,
                linestyle='--',
                marker='o',
                markersize=4,
                color=color,
                linewidth=1.5)

    # Enhanced colorbar with adjusted position
    sm2 = plt.cm.ScalarMappable(cmap='viridis',
                               norm=plt.Normalize(vmin=min(sigma_range), vmax=max(sigma_range)))
    sm2.set_array([])
    cbar2 = plt.colorbar(sm2, ax=ax2, shrink=0.8, aspect=20, pad=0.1)
    cbar2.set_label('Volatility (σ)', fontsize=12, labelpad=10)

    # Enhanced labels with adjusted positions
    ax2.set_xlabel('Time (Years from 2011)', labelpad=15)
    ax2.set_ylabel('σ Value', labelpad=15)
    ax2.set_zlabel('Tourist Numbers', labelpad=5)
    ax2.set_title('(b) Sensitivity Analysis of Volatility (σ)', pad=20)

    # Optimized view angle
    ax2.view_init(elev=25, azim=-60)

    # Add super title with adjusted position
    plt.suptitle('Sensitivity Analysis of Tourist Numbers',
                fontsize=18,
                y=1.02)

    # Optimize layout with adjusted parameters
    plt.tight_layout(rect=[0, 0, 1, 0.95], pad=2.0, w_pad=1.0)

    return fig

# Function to save the figure in publication-quality formats
def save_publication_figure(fig, filename_base):
    """
    Save the figure in multiple publication-ready formats
    """
    # Save as PDF (vector format, best for publication)
    fig.savefig(f'{filename_base}.pdf',
                dpi=300,
                bbox_inches='tight',
                format='pdf')




# Run historical simulation and display results
current_time = "2025-01-24 14:18:42"
print(f"\nAnalysis Time: {current_time} UTC")
print(f"Analysis User: csuce2202")
print("\nStarting simulation analysis...")

# Historical simulation
V, time, mean_path, percentile_5, percentile_95 = simulate_tourist_sde()
historical_fig = plot_historical_simulation(V, time, mean_path, percentile_5, percentile_95, df)

# Calculate metrics for historical simulation
simulated_years = np.floor(time + 2011)
metrics = calculate_error_metrics(df, simulated_years, mean_path)

print("\nHistorical Model Evaluation Metrics:")
print(f"MAPE: {metrics['MAPE']:.2f}%")
print(f"RMSE: {metrics['RMSE']:.0f}")
print(f"R² (Coefficient of Determination): {metrics['R2']:.4f}")
print("\nHistorical Yearly Comparison:")
print(metrics['Yearly_Comparison'].round(2).to_string())

# Save historical plot
historical_fig.savefig('tourist_historical_simulation.pdf', format='pdf')

# Future simulation
future_fig, forecast_years, forecast_values = simulate_future()

print("\nFuture Predictions (2023-2028):")
for year, value in zip(forecast_years, forecast_values):
    print(f"Year {int(year)}: {int(value):,} tourists")

# Save future forecast plot
future_fig.savefig('tourist_future_forecast.pdf', format='pdf')

# Create parameter ranges with more values for smoother visualization
mu_range = np.linspace(0.05, 0.15, 10)
sigma_range = np.linspace(0.1, 0.3, 10)

# Create the visualization
fig = create_3d_sensitivity_analysis(None, mu_range, sigma_range)
fig.savefig('sensitivity_analysis.pdf', format='pdf', bbox_inches='tight')

plt.show()