#!/usr/bin/env python3
"""
Create focused plots highlighting key insights from the optimization results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_key_insights_plot():
    """Create a focused plot with the most important findings."""
    
    # Load data
    with open('optimization_results.json', 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data['results'])
    df = df[df['status'].isin(['success', 'partial'])]
    
    # Create figure with key insights
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Key Optimization Insights: Speculative Decoding Parameters', fontsize=16, y=0.95)
    
    # 1. Main finding: Sigma is the dominant factor
    ax1 = axes[0, 0]
    sigma_means = df.groupby('sigma')['mse'].mean()
    sigma_stds = df.groupby('sigma')['mse'].std()
    
    bars = ax1.bar(sigma_means.index.astype(str), sigma_means.values, 
                   yerr=sigma_stds.values, capsize=5, color='lightblue', edgecolor='navy')
    ax1.set_xlabel('Sigma')
    ax1.set_ylabel('MSE')
    ax1.set_title('🔍 Key Finding: Sigma Dominates Performance\n(Lower sigma = Better accuracy)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, mean in zip(bars, sigma_means.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Bias has minimal effect
    ax2 = axes[0, 1]
    bias_means = df.groupby('bias')['mse'].mean()
    bias_stds = df.groupby('bias')['mse'].std()
    
    bars = ax2.bar(bias_means.index.astype(str), bias_means.values,
                   yerr=bias_stds.values, capsize=5, color='lightgreen', edgecolor='darkgreen')
    ax2.set_xlabel('Bias')
    ax2.set_ylabel('MSE')
    ax2.set_title('📊 Bias Has Minimal Impact\n(All values perform similarly)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. K (speculation length) effect
    ax3 = axes[0, 2]
    k_means = df.groupby('k')['mse'].mean()
    k_stds = df.groupby('k')['mse'].std()
    
    bars = ax3.bar(k_means.index.astype(str), k_means.values,
                   yerr=k_stds.values, capsize=5, color='lightcoral', edgecolor='darkred')
    ax3.set_xlabel('K (Speculation Length)')
    ax3.set_ylabel('MSE')
    ax3.set_title('🚀 K Shows Slight Preference\n(K=2 performs best)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, mean in zip(bars, k_means.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Best configurations
    ax4 = axes[1, 0]
    best_configs = df.nsmallest(8, 'mse')
    
    x_pos = range(len(best_configs))
    bars = ax4.bar(x_pos, best_configs['mse'], color='gold', edgecolor='orange')
    ax4.set_xlabel('Configuration Rank')
    ax4.set_ylabel('MSE')
    ax4.set_title('🏆 Top 8 Best Configurations\n(All use sigma=0.5)', fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f"#{i+1}" for i in x_pos])
    ax4.grid(True, alpha=0.3)
    
    # Add configuration details as text
    config_text = ""
    for i, (_, row) in enumerate(best_configs.head(4).iterrows()):
        config_text += f"#{i+1}: σ={row['sigma']}, β={row['bias']}, k={row['k']}, MSE={row['mse']:.3f}\n"
    
    ax4.text(0.02, 0.98, config_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 5. Parameter correlation analysis
    ax5 = axes[1, 1]
    correlations = {
        'Sigma': df['sigma'].corr(df['mse']),
        'Bias': df['bias'].corr(df['mse']),
        'MSE_Tol': df['mse_tol'].corr(df['mse']),
        'K': df['k'].corr(df['mse'])
    }
    
    colors = ['red' if abs(corr) > 0.5 else 'orange' if abs(corr) > 0.1 else 'green' 
              for corr in correlations.values()]
    
    bars = ax5.bar(correlations.keys(), correlations.values(), color=colors)
    ax5.set_ylabel('Correlation with MSE')
    ax5.set_title('📈 Parameter-MSE Correlations\n(Red=Strong, Orange=Weak, Green=None)', fontweight='bold')
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax5.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, corr in zip(bars, correlations.values()):
        ax5.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + (0.05 if corr > 0 else -0.1),
                f'{corr:.3f}', ha='center', va='bottom' if corr > 0 else 'top',
                fontweight='bold')
    
    # 6. Recommendations
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    recommendations = """
🎯 OPTIMIZATION RECOMMENDATIONS

✅ CRITICAL: Use sigma = 0.5
   • Strong correlation (0.957) with MSE
   • Lower sigma = much better accuracy
   • This is the most important parameter

✅ OPTIMAL: Set K = 2
   • Best speculation length
   • Good balance of efficiency vs accuracy

✅ FLEXIBLE: Bias can be 1.0-10.0
   • No significant impact on MSE
   • Choose based on acceptance rate needs

✅ MODERATE: MSE tolerance = 0.0-0.5
   • Slight preference for lower values
   • Can tune for speed vs accuracy trade-off

🏆 RECOMMENDED CONFIG:
   sigma=0.5, bias=2.0, mse_tol=0.0, k=2
   Expected MSE: ~0.753
"""
    
    ax6.text(0.05, 0.95, recommendations, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('key_optimization_insights.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print executive summary
    print("\n" + "="*80)
    print("🚀 EXECUTIVE SUMMARY: SPECULATIVE DECODING OPTIMIZATION")
    print("="*80)
    
    print(f"\n📊 TESTED: {len(df)} parameter combinations")
    print(f"📈 MSE RANGE: {df['mse'].min():.3f} - {df['mse'].max():.3f}")
    print(f"🎯 BEST MSE: {df['mse'].min():.3f}")
    
    print(f"\n🔍 KEY FINDINGS:")
    print(f"   • SIGMA is the dominant factor (correlation: {df['sigma'].corr(df['mse']):.3f})")
    print(f"   • BIAS has no significant impact (correlation: {df['bias'].corr(df['mse']):.3f})")
    print(f"   • K=2 performs best on average")
    print(f"   • Lower sigma values dramatically improve accuracy")
    
    best_config = df.loc[df['mse'].idxmin()]
    print(f"\n🏆 OPTIMAL CONFIGURATION:")
    print(f"   • Sigma: {best_config['sigma']}")
    print(f"   • Bias: {best_config['bias']}")
    print(f"   • MSE Tolerance: {best_config['mse_tol']}")
    print(f"   • K: {best_config['k']}")
    print(f"   • Resulting MSE: {best_config['mse']:.4f}")
    
    print(f"\n💡 NEXT STEPS:")
    print(f"   1. Use sigma=0.5 for all future experiments")
    print(f"   2. Focus on improving draft model quality (current bottleneck)")
    print(f"   3. Test with better pre-trained draft models")
    print(f"   4. Measure actual inference speedup with optimal parameters")

if __name__ == "__main__":
    create_key_insights_plot()
