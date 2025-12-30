#!/usr/bin/env python3
"""
03_correlation_analysis.py

Analyze the relationship between comprehension and generation resistance.
Reproduces the main RQ1 finding: ρ = 0.21 (weak correlation).
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the combined results dataset."""
    data_path = Path(__file__).parent.parent / 'data' / 'combined_results.csv'
    return pd.read_csv(data_path)

def spearman_correlation(df):
    """
    Calculate Spearman correlation between comprehension and resistance.
    """
    print("\n" + "=" * 60)
    print("SPEARMAN CORRELATION ANALYSIS")
    print("=" * 60)
    
    rho, p = stats.spearmanr(df['comprehension'], df['resistance'])
    
    print(f"\nSpearman ρ = {rho:.3f}")
    print(f"P-value = {p:.6f}")
    print(f"ρ² (variance explained) = {rho**2:.1%}")
    
    # Confidence interval via bootstrap
    n_bootstrap = 10000
    rhos = []
    for _ in range(n_bootstrap):
        sample = df.sample(n=len(df), replace=True)
        r, _ = stats.spearmanr(sample['comprehension'], sample['resistance'])
        rhos.append(r)
    
    ci_lower = np.percentile(rhos, 2.5)
    ci_upper = np.percentile(rhos, 97.5)
    
    print(f"95% CI (bootstrap): [{ci_lower:.3f}, {ci_upper:.3f}]")
    
    return {'rho': rho, 'p': p, 'ci_lower': ci_lower, 'ci_upper': ci_upper}

def pearson_correlation(df):
    """
    Calculate Pearson correlation for comparison.
    """
    print("\n" + "-" * 40)
    print("PEARSON CORRELATION (for comparison)")
    print("-" * 40)
    
    r, p = stats.pearsonr(df['comprehension'], df['resistance'])
    
    print(f"\nPearson r = {r:.3f}")
    print(f"P-value = {p:.6f}")
    
    return {'r': r, 'p': p}

def correlation_by_subgroup(df):
    """
    Calculate correlations within subgroups.
    """
    print("\n" + "=" * 60)
    print("CORRELATION BY SUBGROUP")
    print("=" * 60)
    
    results = {}
    
    # By model type
    print("\nBy Model Type:")
    print(f"{'Type':<15} {'n':>6} {'ρ':>8} {'p':>10}")
    print("-" * 45)
    
    for mtype in df['model_type'].unique():
        subset = df[df['model_type'] == mtype]
        if len(subset) > 10:
            rho, p = stats.spearmanr(subset['comprehension'], subset['resistance'])
            print(f"{mtype:<15} {len(subset):>6} {rho:>8.3f} {p:>10.4f}")
            results[f'type_{mtype}'] = {'rho': rho, 'p': p, 'n': len(subset)}
    
    # By vulnerability
    print("\nBy Vulnerability Type:")
    print(f"{'CWE':<15} {'n':>6} {'ρ':>8} {'p':>10}")
    print("-" * 45)
    
    for cwe in sorted(df['cwe'].unique()):
        subset = df[df['cwe'] == cwe]
        rho, p = stats.spearmanr(subset['comprehension'], subset['resistance'])
        print(f"{cwe:<15} {len(subset):>6} {rho:>8.3f} {p:>10.4f}")
        results[f'cwe_{cwe}'] = {'rho': rho, 'p': p, 'n': len(subset)}
    
    # Open-source only (to check for ceiling effects)
    print("\nOpen-Source Models Only:")
    open_source = df[df['model_type'].isin(['Code', 'General', 'Safety', 'MoE'])]
    rho, p = stats.spearmanr(open_source['comprehension'], open_source['resistance'])
    print(f"n={len(open_source)}, ρ={rho:.3f}, p={p:.4f}")
    results['open_source'] = {'rho': rho, 'p': p, 'n': len(open_source)}
    
    return results

def regression_analysis(df):
    """
    Linear regression to quantify relationship.
    """
    print("\n" + "=" * 60)
    print("LINEAR REGRESSION ANALYSIS")
    print("=" * 60)
    
    from scipy.stats import linregress
    
    slope, intercept, r, p, se = linregress(df['comprehension'], df['resistance'])
    
    print(f"\nResistance = {intercept:.3f} + {slope:.3f} × Comprehension")
    print(f"R² = {r**2:.3f}")
    print(f"Standard Error = {se:.4f}")
    print(f"P-value = {p:.6f}")
    
    print("\nInterpretation:")
    print(f"  A 10% increase in comprehension is associated with")
    print(f"  a {slope * 0.1:.1%} change in resistance")
    
    return {'slope': slope, 'intercept': intercept, 'r2': r**2, 'p': p}

def sensitivity_analysis(df):
    """
    Test alternative resistance formulations.
    """
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS: Alternative Formulations")
    print("=" * 60)
    
    formulations = {
        'Main (breach_round-1)/N': df['resistance'],
        'Binary (breached/not)': (df['breach_round'] == 0).astype(float),
        'Linear decay': 1 - df['breach_round'] / 10,
    }
    
    print(f"\n{'Formulation':<30} {'ρ':>8} {'p':>10}")
    print("-" * 50)
    
    for name, resist in formulations.items():
        rho, p = stats.spearmanr(df['comprehension'], resist)
        print(f"{name:<30} {rho:>8.3f} {p:>10.4f}")

def main():
    print("Loading data...")
    df = load_data()
    
    # Main correlation
    spearman = spearman_correlation(df)
    pearson = pearson_correlation(df)
    
    # Subgroup analysis
    subgroups = correlation_by_subgroup(df)
    
    # Regression
    regression = regression_analysis(df)
    
    # Sensitivity
    sensitivity_analysis(df)
    
    # Summary box (matching paper)
    print("\n" + "=" * 60)
    print("RQ1 ANSWER (from paper)")
    print("=" * 60)
    print(f"""
    76.0% comprehension vs 92.4% resistance with weak correlation
    (ρ={spearman['rho']:.2f}, ρ²={spearman['rho']**2:.0%}). 
    
    Comprehension explains only {spearman['rho']**2:.0%} of resistance variance.
    """)

if __name__ == '__main__':
    main()
