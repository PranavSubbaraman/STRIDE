#!/usr/bin/env python3
"""
Plot optimization results for speculative decoding parameters.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

def load_results():
    """Load optimization results from JSON file."""
    with open('optimization_results.json', 'r') as f:
        data = json.load(f)
    
    results = data['results']
    baseline_time = data.get('baseline_time')
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Filter out failed results
    df = df[df['status'].isin(['success', 'partial'])]
    
    print(f"Loaded {len(df)} results")
    print(f"Status distribution: {df['status'].value_counts().to_dict()}")
    
    return df, baseline_time

def plot_parameter_effects(df):
    """Plot how different parameters affect MSE and acceptance rate."""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. MSE vs Sigma for different bias values
    ax1 = fig.add_subplot(gs[0, 0])
    for bias in sorted(df['bias'].unique()):
        subset = df[df['bias'] == bias]
        if len(subset) > 0:
            grouped = subset.groupby('sigma')['mse'].mean()
            ax1.plot(grouped.index, grouped.values, marker='o', label=f'bias={bias}')
    ax1.set_xlabel('Sigma')
    ax1.set_ylabel('MSE')
    ax1.set_title('MSE vs Sigma (by Bias)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. MSE vs Bias for different sigma values
    ax2 = fig.add_subplot(gs[0, 1])
    for sigma in sorted(df['sigma'].unique()):
        subset = df[df['sigma'] == sigma]
        if len(subset) > 0:
            grouped = subset.groupby('bias')['mse'].mean()
            ax2.plot(grouped.index, grouped.values, marker='o', label=f'sigma={sigma}')
    ax2.set_xlabel('Bias')
    ax2.set_ylabel('MSE')
    ax2.set_title('MSE vs Bias (by Sigma)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. MSE vs MSE Tolerance
    ax3 = fig.add_subplot(gs[0, 2])
    grouped = df.groupby('mse_tol')['mse'].mean()
    ax3.bar(grouped.index.astype(str), grouped.values)
    ax3.set_xlabel('MSE Tolerance')
    ax3.set_ylabel('MSE')
    ax3.set_title('MSE vs MSE Tolerance')
    ax3.grid(True, alpha=0.3)
    
    # 4. MSE vs K
    ax4 = fig.add_subplot(gs[0, 3])
    grouped = df.groupby('k')['mse'].mean()
    ax4.bar(grouped.index.astype(str), grouped.values)
    ax4.set_xlabel('K (speculation length)')
    ax4.set_ylabel('MSE')
    ax4.set_title('MSE vs K')
    ax4.grid(True, alpha=0.3)
    
    # 5. Heatmap: MSE vs Sigma and Bias
    ax5 = fig.add_subplot(gs[1, :2])
    pivot_mse = df.groupby(['sigma', 'bias'])['mse'].mean().unstack()
    sns.heatmap(pivot_mse, annot=True, fmt='.3f', cmap='viridis_r', ax=ax5)
    ax5.set_title('MSE Heatmap: Sigma vs Bias')
    ax5.set_xlabel('Bias')
    ax5.set_ylabel('Sigma')
    
    # 6. Distribution of MSE values
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(df['mse'], bins=30, alpha=0.7, edgecolor='black')
    ax6.set_xlabel('MSE')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Distribution of MSE Values')
    ax6.grid(True, alpha=0.3)
    
    # 7. MSE vs MAE correlation
    ax7 = fig.add_subplot(gs[1, 3])
    ax7.scatter(df['mse'], df['mae'], alpha=0.6)
    ax7.set_xlabel('MSE')
    ax7.set_ylabel('MAE')
    ax7.set_title('MSE vs MAE Correlation')
    
    # Add correlation coefficient
    corr = df['mse'].corr(df['mae'])
    ax7.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax7.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax7.grid(True, alpha=0.3)
    
    # 8. Parameter combination performance
    ax8 = fig.add_subplot(gs[2, :])
    
    # Create a combined parameter string for x-axis
    df['param_combo'] = df.apply(lambda x: f"σ={x['sigma']}, β={x['bias']}, t={x['mse_tol']}, k={x['k']}", axis=1)
    
    # Sort by MSE and take top 20 best and worst
    df_sorted = df.sort_values('mse')
    top_20 = df_sorted.head(20)
    bottom_20 = df_sorted.tail(20)
    
    # Plot best configurations
    x_pos = range(len(top_20))
    bars = ax8.bar(x_pos, top_20['mse'], color='green', alpha=0.7, label='Best 20 configs')
    ax8.set_xlabel('Configuration')
    ax8.set_ylabel('MSE')
    ax8.set_title('Top 20 Best Parameter Combinations (by MSE)')
    ax8.set_xticks(x_pos[::2])  # Show every other label to avoid crowding
    ax8.set_xticklabels([combo.replace(', ', '\\n') for combo in top_20['param_combo'].iloc[::2]], 
                        rotation=45, ha='right', fontsize=8)
    ax8.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mse) in enumerate(zip(bars, top_20['mse'])):
        if i % 2 == 0:  # Only label every other bar to avoid crowding
            ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{mse:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 9. Parameter sensitivity analysis
    ax9 = fig.add_subplot(gs[3, :2])
    
    # Calculate coefficient of variation (std/mean) for each parameter
    param_sensitivity = {}
    for param in ['sigma', 'bias', 'mse_tol', 'k']:
        param_means = df.groupby(param)['mse'].mean()
        param_sensitivity[param] = param_means.std() / param_means.mean()
    
    params = list(param_sensitivity.keys())
    sensitivities = list(param_sensitivity.values())
    
    bars = ax9.bar(params, sensitivities, color=['blue', 'orange', 'green', 'red'])
    ax9.set_xlabel('Parameter')
    ax9.set_ylabel('Coefficient of Variation (std/mean)')
    ax9.set_title('Parameter Sensitivity Analysis')
    ax9.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, sens in zip(bars, sensitivities):
        ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{sens:.3f}', ha='center', va='bottom')
    
    # 10. Best configurations table
    ax10 = fig.add_subplot(gs[3, 2:])
    ax10.axis('tight')
    ax10.axis('off')
    
    # Create table data
    best_configs = df_sorted.head(10)[['sigma', 'bias', 'mse_tol', 'k', 'mse', 'mae']].round(3)
    
    table = ax10.table(cellText=best_configs.values,
                      colLabels=best_configs.columns,
                      cellLoc='center',
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Color code the table
    for i in range(len(best_configs)):
        for j in range(len(best_configs.columns)):
            if j == 4:  # MSE column
                if i < 3:  # Top 3
                    table[(i+1, j)].set_facecolor('#90EE90')  # Light green
                elif i < 6:  # Next 3
                    table[(i+1, j)].set_facecolor('#FFFFE0')  # Light yellow
    
    ax10.set_title('Top 10 Best Configurations', pad=20)
    
    plt.suptitle('Speculative Decoding Parameter Optimization Results', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig('optimization_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary_stats(df):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("OPTIMIZATION SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nTotal configurations tested: {len(df)}")
    print(f"Parameter ranges:")
    print(f"  Sigma: {df['sigma'].min()} - {df['sigma'].max()}")
    print(f"  Bias: {df['bias'].min()} - {df['bias'].max()}")
    print(f"  MSE Tolerance: {df['mse_tol'].min()} - {df['mse_tol'].max()}")
    print(f"  K: {df['k'].min()} - {df['k'].max()}")
    
    print(f"\nMSE Statistics:")
    print(f"  Best (lowest): {df['mse'].min():.4f}")
    print(f"  Worst (highest): {df['mse'].max():.4f}")
    print(f"  Mean: {df['mse'].mean():.4f}")
    print(f"  Std: {df['mse'].std():.4f}")
    
    print(f"\nTop 5 configurations by MSE:")
    top_5 = df.nsmallest(5, 'mse')[['sigma', 'bias', 'mse_tol', 'k', 'mse', 'mae']]
    for i, row in top_5.iterrows():
        print(f"  {i+1}. σ={row['sigma']}, β={row['bias']}, tol={row['mse_tol']}, k={row['k']} → MSE={row['mse']:.4f}")
    
    print(f"\nParameter correlations with MSE:")
    for param in ['sigma', 'bias', 'mse_tol', 'k']:
        corr = df[param].corr(df['mse'])
        print(f"  {param}: {corr:.3f}")

def main():
    """Main function to generate all plots and analysis."""
    print("Loading optimization results...")
    df, baseline_time = load_results()
    
    if len(df) == 0:
        print("No valid results found!")
        return
    
    print("Generating plots...")
    plot_parameter_effects(df)
    
    print("Generating summary statistics...")
    print_summary_stats(df)
    
    # Save processed data
    df.to_csv('optimization_results_processed.csv', index=False)
    print(f"\nProcessed data saved to 'optimization_results_processed.csv'")
    print(f"Plots saved to 'optimization_analysis.png'")

if __name__ == "__main__":
    main()
