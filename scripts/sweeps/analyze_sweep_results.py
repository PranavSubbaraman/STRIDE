#!/usr/bin/env python3
"""
Analyze stochastic draft parameter sweep results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

def load_results(csv_path='stochastic_sweep_results.csv'):
    """Load sweep results"""
    df = pd.read_csv(csv_path)
    return df

def print_summary_stats(df):
    """Print summary statistics"""
    print("=" * 80)
    print("PARAMETER SWEEP SUMMARY")
    print("=" * 80)
    print(f"\nTotal configurations tested: {len(df)}")
    print(f"MSE range: [{df['mse'].min():.4f}, {df['mse'].max():.4f}]")
    print(f"MAE range: [{df['mae'].min():.4f}, {df['mae'].max():.4f}]")
    print(f"Acceptance range: [{df['acceptance_pct'].min():.2f}%, {df['acceptance_pct'].max():.2f}%]")
    print(f"Average time per config: {df['total_time_s'].mean():.1f}s")
    
    print("\n" + "=" * 80)
    print("TOP 5 BY ACCEPTANCE RATE")
    print("=" * 80)
    top_accept = df.nlargest(5, 'acceptance_pct')[['sigma', 'bias', 'k', 'mse', 'mae', 'acceptance_pct']]
    print(top_accept.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("TOP 5 BY LOWEST MSE")
    print("=" * 80)
    top_mse = df.nsmallest(5, 'mse')[['sigma', 'bias', 'k', 'mse', 'mae', 'acceptance_pct']]
    print(top_mse.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("BEST BALANCED (Acceptance >= 70%, Lowest MSE)")
    print("=" * 80)
    balanced = df[df['acceptance_pct'] >= 70.0].nsmallest(10, 'mse')[['sigma', 'bias', 'k', 'mse', 'mae', 'acceptance_pct']]
    if len(balanced) > 0:
        print(balanced.to_string(index=False))
    else:
        print("No configurations with acceptance >= 70%")
    
    print("\n" + "=" * 80)
    print("PARAMETER CORRELATIONS")
    print("=" * 80)
    corr = df[['sigma', 'bias', 'k', 'mse', 'acceptance_pct']].corr()
    print("\nCorrelation with MSE:")
    print(corr['mse'].sort_values(ascending=False))
    print("\nCorrelation with Acceptance:")
    print(corr['acceptance_pct'].sort_values(ascending=False))

def plot_results(df, output_dir='sweep_plots'):
    """Generate visualization plots"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Acceptance vs MSE scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df['acceptance_pct'], df['mse'], 
                        c=df['k'], cmap='viridis', s=100, alpha=0.6)
    ax.set_xlabel('Acceptance Rate (%)', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('Acceptance Rate vs MSE (colored by K)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='K (speculation length)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/acceptance_vs_mse.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/acceptance_vs_mse.png")
    plt.close()
    
    # 2. Heatmap: Acceptance by sigma and bias (averaged over K)
    pivot_accept = df.groupby(['sigma', 'bias'])['acceptance_pct'].mean().reset_index()
    pivot_table_accept = pivot_accept.pivot(index='bias', columns='sigma', values='acceptance_pct')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot_table_accept, annot=True, fmt='.1f', cmap='YlGnBu', 
                cbar_kws={'label': 'Acceptance %'}, ax=ax)
    ax.set_title('Average Acceptance Rate by Sigma and Bias', fontsize=14, fontweight='bold')
    ax.set_xlabel('Sigma (target variance)', fontsize=12)
    ax.set_ylabel('Bias', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/acceptance_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/acceptance_heatmap.png")
    plt.close()
    
    # 3. Heatmap: MSE by sigma and bias (averaged over K)
    pivot_mse = df.groupby(['sigma', 'bias'])['mse'].mean().reset_index()
    pivot_table_mse = pivot_mse.pivot(index='bias', columns='sigma', values='mse')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot_table_mse, annot=True, fmt='.4f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'MSE'}, ax=ax)
    ax.set_title('Average MSE by Sigma and Bias', fontsize=14, fontweight='bold')
    ax.set_xlabel('Sigma (target variance)', fontsize=12)
    ax.set_ylabel('Bias', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/mse_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/mse_heatmap.png")
    plt.close()
    
    # 4. Line plot: Effect of K
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    k_stats = df.groupby('k').agg({
        'acceptance_pct': ['mean', 'std'],
        'mse': ['mean', 'std']
    }).reset_index()
    k_stats.columns = ['k', 'accept_mean', 'accept_std', 'mse_mean', 'mse_std']
    
    ax1.errorbar(k_stats['k'], k_stats['accept_mean'], yerr=k_stats['accept_std'],
                marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
    ax1.set_xlabel('K (speculation length)', fontsize=12)
    ax1.set_ylabel('Acceptance Rate (%)', fontsize=12)
    ax1.set_title('Effect of K on Acceptance', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.errorbar(k_stats['k'], k_stats['mse_mean'], yerr=k_stats['mse_std'],
                marker='s', capsize=5, capthick=2, linewidth=2, markersize=8, color='coral')
    ax2.set_xlabel('K (speculation length)', fontsize=12)
    ax2.set_ylabel('MSE', fontsize=12)
    ax2.set_title('Effect of K on MSE', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/k_effect.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/k_effect.png")
    plt.close()
    
    # 5. Pareto frontier (acceptance vs MSE)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Find Pareto optimal points (maximize acceptance, minimize MSE)
    sorted_df = df.sort_values('acceptance_pct', ascending=False)
    pareto_points = []
    min_mse = float('inf')
    
    for _, row in sorted_df.iterrows():
        if row['mse'] < min_mse:
            pareto_points.append(row)
            min_mse = row['mse']
    
    pareto_df = pd.DataFrame(pareto_points).sort_values('acceptance_pct')
    
    ax.scatter(df['acceptance_pct'], df['mse'], alpha=0.3, s=50, label='All configs')
    ax.scatter(pareto_df['acceptance_pct'], pareto_df['mse'], 
              color='red', s=100, marker='*', label='Pareto optimal', zorder=5)
    ax.plot(pareto_df['acceptance_pct'], pareto_df['mse'], 
           'r--', linewidth=2, alpha=0.5, zorder=4)
    
    ax.set_xlabel('Acceptance Rate (%)', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('Pareto Frontier: Acceptance vs MSE Trade-off', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pareto_frontier.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/pareto_frontier.png")
    plt.close()
    
    print(f"\nAll plots saved to {output_dir}/")

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'stochastic_sweep_results.csv'
    
    print(f"Loading results from: {csv_path}")
    df = load_results(csv_path)
    
    print_summary_stats(df)
    
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    plot_results(df)
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    # Find best configuration
    # Priority: High acceptance (>70%), low MSE
    good_accept = df[df['acceptance_pct'] >= 70.0]
    if len(good_accept) > 0:
        best = good_accept.nsmallest(1, 'mse').iloc[0]
        print(f"\n🏆 RECOMMENDED CONFIGURATION:")
        print(f"   sigma = {best['sigma']}")
        print(f"   bias = {best['bias']}")
        print(f"   k = {int(best['k'])}")
        print(f"   → MSE: {best['mse']:.4f}")
        print(f"   → MAE: {best['mae']:.4f}")
        print(f"   → Acceptance: {best['acceptance_pct']:.2f}%")
    else:
        best = df.nsmallest(1, 'mse').iloc[0]
        print(f"\n🏆 BEST MSE CONFIGURATION (note: lower acceptance):")
        print(f"   sigma = {best['sigma']}")
        print(f"   bias = {best['bias']}")
        print(f"   k = {int(best['k'])}")
        print(f"   → MSE: {best['mse']:.4f}")
        print(f"   → MAE: {best['mae']:.4f}")
        print(f"   → Acceptance: {best['acceptance_pct']:.2f}%")

if __name__ == '__main__':
    main()
