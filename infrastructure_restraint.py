import numpy as np
import pandas as pd
from queue import Queue
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class InfrastructureModel:
    def __init__(self, data, max_capacity=None, service_rate=None):
        self.data = data
        self.data['Visitor Volume Daily'] = self.data['Visitor Volume'] / 365  # Daily visitor volume calculation
        self.max_capacity = max_capacity or self.data['Visitor Volume Daily'].max()  # Maximum capacity per day
        self.service_rate = service_rate or self.max_capacity * 1.2
        self.queue = Queue()

        # Calculate historical parameters
        self.calculate_historical_params()

    def calculate_historical_params(self):
        """Calculate key parameters from historical data"""
        self.daily_metrics = {
            'peak_arrivals': self.data['Visitor Volume Daily'].max(),  # Peak daily arrivals
            'avg_arrivals': self.data['Visitor Volume Daily'].mean(),  # Average daily arrivals
            'std_arrivals': self.data['Visitor Volume Daily'].std()  # Standard deviation of daily arrivals
        }

        self.seasonal_metrics = {
            'summer_ratio': self.data['Summer'].mean() / self.data['Visitor Volume'].mean(),
            'winter_ratio': self.data['Winter'].mean() / self.data['Visitor Volume'].mean()
        }

        # Calculate transition probabilities (remains the same)
        self.transition_matrix = {
            'peak': {'peak': 0.8, 'off_peak': 0.2},
            'off_peak': {'peak': 0.3, 'off_peak': 0.7}
        }

    def mm1_queue_metrics(self, arrival_rate):
        """M/M/1 queue system metrics calculation"""
        rho = arrival_rate / self.service_rate
        if rho >= 1:
            return float('inf'), float('inf')

        L = rho / (1 - rho)  # Average number of people in the system
        W = 1 / (self.service_rate - arrival_rate)  # Average waiting time
        return L, W

    def markov_season_transition(self, days=365):
        """Markov chain model for seasonal transitions"""
        states = ['peak', 'off_peak']
        current_state = 'peak' if random.random() < self.seasonal_metrics['summer_ratio'] else 'off_peak'
        state_history = [current_state]

        for _ in range(days):
            next_prob = random.random()
            cum_prob = 0
            for next_state in states:
                cum_prob += self.transition_matrix[current_state][next_state]
                if next_prob <= cum_prob:
                    current_state = next_state
                    break
            state_history.append(current_state)

        return state_history

    def generate_daily_arrivals(self, season_pattern):
        """Generate daily tourist arrivals"""
        daily_arrivals = []
        for season in season_pattern:
            if season == 'peak':
                mean = self.daily_metrics['peak_arrivals']
                std = self.daily_metrics['std_arrivals']
            else:
                mean = self.daily_metrics['avg_arrivals']
                std = self.daily_metrics['std_arrivals'] * 0.5

            arrivals = np.random.normal(mean, std)
            daily_arrivals.append(max(0, arrivals))

        return daily_arrivals

    def simulate_tourist_flow(self, simulation_days=365):
        """Simulate tourist flow and infrastructure pressure for 5 years"""
        # Create a list to store the daily arrivals
        daily_arrivals = []

        # Calculate the number of days in each year
        days_per_year = 365
        num_years = simulation_days // days_per_year

        # Loop through each year and simulate arrivals
        for year in range(1, num_years + 1):
            # Get the visitor volume for the current year
            annual_volume = annual_visitor_volume[year]

            # Calculate the daily visitor volume for this year
            daily_volume = annual_volume / days_per_year

            # Generate seasonal pattern (Markov chain)
            season_pattern = self.markov_season_transition(days=days_per_year)

            # Generate daily arrivals based on the season pattern for this year
            yearly_arrivals = self.generate_daily_arrivals(season_pattern)

            # Append this year's results to the daily_arrivals list
            daily_arrivals.extend(yearly_arrivals)

        # Calculate the queue and wait time for each day
        daily_metrics = []
        for day in range(simulation_days):
            arrival_rate = daily_arrivals[day]
            L, W = self.mm1_queue_metrics(arrival_rate)
            pressure = arrival_rate / self.max_capacity

            daily_metrics.append({
                'day': day,
                'season': season_pattern[day % days_per_year],  # Cycle through seasons for each year
                'arrivals': arrival_rate,
                'queue_length': L,
                'wait_time': W,
                'pressure': pressure
            })

        return pd.DataFrame(daily_metrics)

    def plot_results_enhanced(self, results):
        """Plot simulation results with enhanced visual style"""
        # Define a more modern style
        plt.style.use('seaborn-whitegrid')

        # Set up font and size for titles and labels
        font = {'family': 'sans-serif', 'weight': 'bold', 'size': 12}
        plt.rc('font', **font)

        # Create the plot figure and axes
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Set a common title for the figure
        fig.suptitle('Tourist Flow Simulation Results (5 years)', fontsize=16, fontweight='bold', y=1.03)

        # Daily tourist arrivals
        axes[0, 0].plot(results['day'], results['arrivals'], color='dodgerblue', linewidth=2)
        axes[0, 0].set_title('Daily Tourist Arrivals', fontsize=14)
        axes[0, 0].set_xlabel('Day')
        axes[0, 0].set_ylabel('Arrivals')
        axes[0, 0].grid(True, which='both', linestyle='--', linewidth=0.5)

        # Wait time
        axes[0, 1].plot(results['day'], results['wait_time'], color='orange', linewidth=2)
        axes[0, 1].set_title('Wait Time (Hours)', fontsize=14)
        axes[0, 1].set_xlabel('Day')
        axes[0, 1].set_ylabel('Wait Time (hrs)')
        axes[0, 1].grid(True, which='both', linestyle='--', linewidth=0.5)

        # Queue length
        axes[1, 0].plot(results['day'], results['queue_length'], color='forestgreen', linewidth=2)
        axes[1, 0].set_title('Queue Length', fontsize=14)
        axes[1, 0].set_xlabel('Day')
        axes[1, 0].set_ylabel('Queue Length')
        axes[1, 0].grid(True, which='both', linestyle='--', linewidth=0.5)

        # Infrastructure pressure
        axes[1, 1].plot(results['day'], results['pressure'], color='tomato', linewidth=2)
        axes[1, 1].set_title('Infrastructure Pressure', fontsize=14)
        axes[1, 1].set_xlabel('Day')
        axes[1, 1].set_ylabel('Pressure Index')
        axes[1, 1].grid(True, which='both', linestyle='--', linewidth=0.5)

        # Tight layout to prevent overlap
        plt.tight_layout()

        # Save the plot as a PDF
        plt.savefig('tourist_flow_simulation_enhanced.pdf', format='pdf')

        # Show the plot
        plt.show()

        # Print statistics summary
        print("\nSimulation Results Summary:")
        print(results.describe())
        print("\nSeason Distribution:")
        print(results['season'].value_counts(normalize=True))


def main():
    # Define future 5 years' predicted visitor volume (you provided V1 to V5)
    global annual_visitor_volume
    annual_visitor_volume = {
        1: 1953540,
        2: 2035840,
        3: 2064580,
        4: 2105400,
        5: 2204565
    }

    # Input data manually (replace with actual values as needed)
    data_dict = {
        'Winter': [
            144022, 157416, 155233, 161074, 168744, 186149, 186798, 189394, 190574,
            188500, 262580
        ],
        'Summer': [
            918533, 936116, 999365, 979187, 1050224, 1095950, 1136543, 1195544, 1305700,
            1167000, 1670000
        ],
        'Total': [
            1062555, 1093532, 1154598, 1140261, 1218968, 1282099, 1323341, 1384938, 1496274,
            1355500, 1932580
        ]
    }

    # Convert data into a DataFrame
    data = pd.DataFrame(data_dict)

    # Calculate 'Visitor Volume' (sum of Winter and Summer)
    data['Visitor Volume'] = data['Winter'] + data['Summer']

    # Calculate daily visitor volume by dividing by 365
    data['Visitor Volume Daily'] = data['Visitor Volume'] / 365

    # Initialize model with input data
    model = InfrastructureModel(data)

    # Run simulation for 5 years
    results = model.simulate_tourist_flow(simulation_days=365 * 5)

    # Display results
    model.plot_results_enhanced(results)


if __name__ == "__main__":
    main()
