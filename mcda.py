import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator

# Create data with corresponding years
years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2022, 2023, 2024, 2025, 2026, 2027, 2028]
data = {
    'Year': years,
    'V(t)': [636978.69,777021.1,927558.13,1091795.04,1251974.87,1405688.06,1575884,1723066.96,1871075.21,2175271.75,2228074.61,1969919,2024811,2058573,2105485,2141251],
    'I(t)': [0.964,0.9281,0.9091,0.8987,0.8997,0.9243,0.9653,1.0218,1.1049,1.1792,1.3242,1.6262,1.7802,1.947,2.1261,2.317],
    'P(t)': [0.7,0.64,0.58,0.52,0.46,0.4,0.34,0.28,0.22,0.03,-0.03,-0.1,-0.13,-0.24,-0.24,-0.37],
    'R(t)': [924820000,1115280000,1305740000,1496190000,1686650000,1877110000,2067560000,2258020000,2448480000,3019850000,3210300000,2973600000,3217350000,4271660000,3554640000,4721470000]
}

df = pd.DataFrame(data)

# Constraints
V_max = 2000000
P_min = -1
I_max = 2.0

# Filter data
df_filtered = df[(df['V(t)'] < V_max) & (df['P(t)'] > P_min) & (df['I(t)'] < I_max)].copy()

# Normalize data
df_normalized = df_filtered.copy()
for column in ['V(t)', 'I(t)', 'P(t)', 'R(t)']:
    df_normalized[column] = (df_filtered[column] - df_filtered[column].min()) / (df_filtered[column].max() - df_filtered[column].min())

# AHP weights calculation
pairwise_comparison_matrix = np.array([
    # V(t)       I(t)       P(t)       R(t)
    [1,         1/5,       1/3,       3],    # 游客数量 vs 其他
    [5,         1,          3,         7],    # 环境影响 vs 其他
    [3,         1/3,       1,         5],    # 公众意见 vs 其他
    [1/3,       1/7,       1/5,       1]     # 旅游业收入 vs 其他
])

column_sums = pairwise_comparison_matrix.sum(axis=0)
normalized_matrix = pairwise_comparison_matrix / column_sums
weights_ahp = normalized_matrix.mean(axis=1)

# TOPSIS calculation
df_weighted = df_normalized[['V(t)', 'I(t)', 'P(t)', 'R(t)']] * weights_ahp
ideal_solution = df_weighted.max()
negative_ideal_solution = df_weighted.min()

distance_to_ideal = np.sqrt(((df_weighted - ideal_solution) ** 2).sum(axis=1))
distance_to_negative_ideal = np.sqrt(((df_weighted - negative_ideal_solution) ** 2).sum(axis=1))
closeness = distance_to_negative_ideal / (distance_to_ideal + distance_to_negative_ideal)

df_filtered['Closeness'] = closeness
df_filtered_sorted = df_filtered.sort_values(by='Closeness', ascending=False)

# Plotting
plt.style.use('seaborn-paper')
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

fig, ax = plt.subplots(figsize=(16, 8), dpi=300)

closeness_values = df_filtered_sorted['Closeness'][::-1]
plot_years = df_filtered_sorted['Year'][::-1]

bars = ax.barh(range(len(plot_years)),
               closeness_values,
               height=0.6,
               color='#2e86de',
               alpha=0.8,
               edgecolor='#2573c1',
               linewidth=1)

ax.grid(True, axis='x', linestyle='--', alpha=0.3, zorder=0)
ax.set_axisbelow(True)

ax.set_title('Comprehensive Evaluation Results Based on TOPSIS-AHP',
             pad=20, fontsize=14, fontweight='bold')
ax.set_xlabel('Relative Closeness Coefficient', fontsize=12, labelpad=10)
ax.set_ylabel('Year', fontsize=12, labelpad=10)

ax.set_yticks(range(len(plot_years)))
ax.set_yticklabels(plot_years, fontsize=10)
ax.set_xlim(0, max(closeness_values) * 1.15)
ax.xaxis.set_minor_locator(AutoMinorLocator())

for i, v in enumerate(closeness_values):
    ax.text(v + 0.01, i, f'{v:.4f}',
            va='center', ha='left', fontsize=9,
            color='#34495e', fontweight='bold')

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.5)
    spine.set_color('#2c3e50')

ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='x', which='minor', bottom=True)

plt.tight_layout()
plt.savefig('TOPSIS_AHP_Results.pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()