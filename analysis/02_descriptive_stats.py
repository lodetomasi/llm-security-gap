#!/usr/bin/env python3
"""
02_descriptive_stats.py

Generate Table 2 (Comprehension vs Generation by Model) and 
Table 3 (Vulnerability-Specific Resistance) from the paper.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_data():
    """Load the combined results dataset."""
    data_path = Path(__file__).parent.parent / 'data' / 'combined_results.csv'
    if not data_path.exists():
        print("Data not found. Run 00_generate_data.py first.")
        exit(1)
    return pd.read_csv(data_path)

def generate_table2(df):
    """
    Generate Table 2: Comprehension vs Generation Resistance by Model
    """
    print("\n" + "=" * 80)
    print("TABLE 2: Comprehension vs. Generation Resistance by Model")
    print("=" * 80)
    
    # Aggregate by model
    model_stats = df.groupby(['model', 'model_type']).agg({
        'comprehension': 'mean',
        'resistance': 'mean'
    }).reset_index()
    
    # Calculate gap and breach rate
    model_stats['gap'] = model_stats['resistance'] - model_stats['comprehension']
    model_stats['breach_rate'] = 1 - model_stats['resistance']
    
    # Sort by breach rate (descending) within type
    model_stats = model_stats.sort_values(['model_type', 'breach_rate'], 
                                           ascending=[True, False])
    
    # Print table
    print(f"\n{'Model':<20} {'Comp.':>8} {'Resist.':>8} {'Gap':>8} {'Breach':>8} {'Type':<12}")
    print("-" * 80)
    
    for _, row in model_stats.iterrows():
        print(f"{row['model']:<20} {row['comprehension']:>7.1%} {row['resistance']:>7.1%} "
              f"{row['gap']:>+7.1%} {row['breach_rate']:>7.1%} {row['model_type']:<12}")
    
    # Print means by type
    print("-" * 80)
    for mtype in ['Code', 'General', 'Safety', 'MoE', 'Proprietary']:
        subset = model_stats[model_stats['model_type'] == mtype]
        if len(subset) > 0:
            print(f"{mtype + ' Mean':<20} {subset['comprehension'].mean():>7.1%} "
                  f"{subset['resistance'].mean():>7.1%} {subset['gap'].mean():>+7.1%} "
                  f"{subset['breach_rate'].mean():>7.1%}")
    
    # Overall mean
    print("-" * 80)
    print(f"{'Overall Mean':<20} {model_stats['comprehension'].mean():>7.1%} "
          f"{model_stats['resistance'].mean():>7.1%} {model_stats['gap'].mean():>+7.1%} "
          f"{model_stats['breach_rate'].mean():>7.1%}")
    
    return model_stats

def generate_table3(df):
    """
    Generate Table 3: Vulnerability-Specific Resistance by Model
    """
    print("\n" + "=" * 100)
    print("TABLE 3: Vulnerability-Specific Resistance (%) by Model")
    print("=" * 100)
    
    # Create pivot table
    pivot = df.pivot_table(
        values='resistance',
        index='model',
        columns='cwe',
        aggfunc='mean'
    ) * 100  # Convert to percentage
    
    # Add mean column
    pivot['Mean'] = pivot.mean(axis=1)
    
    # Add model type
    model_types = df.groupby('model')['model_type'].first()
    pivot['Type'] = model_types
    
    # Sort
    pivot = pivot.sort_values('Mean', ascending=False)
    
    # Print header
    cwes = ['CWE-89', 'CWE-78', 'CWE-79', 'CWE-22', 'CWE-918', 'CWE-611', 'CWE-502', 'CWE-94']
    header = f"{'Model':<20}" + "".join([f"{c.replace('CWE-', ''):>8}" for c in cwes])
    header += f"{'Mean':>8} {'Type':<10}"
    print(f"\n{header}")
    print("-" * 100)
    
    # Print rows
    for model, row in pivot.iterrows():
        line = f"{model:<20}"
        for cwe in cwes:
            val = row.get(cwe, 0)
            line += f"{val:>7.1f}%"
        line += f"{row['Mean']:>7.1f}% {row['Type']:<10}"
        print(line)
    
    # Print vulnerability means
    print("-" * 100)
    mean_line = f"{'Vuln Mean':<20}"
    for cwe in cwes:
        mean_line += f"{pivot[cwe].mean():>7.1f}%"
    mean_line += f"{pivot['Mean'].mean():>7.1f}%"
    print(mean_line)
    
    return pivot

def generate_summary_stats(df):
    """
    Generate overall summary statistics.
    """
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    stats = {
        'Comprehension': {
            'Mean': df['comprehension'].mean(),
            'SD': df['comprehension'].std(),
            'Min': df['comprehension'].min(),
            'Max': df['comprehension'].max(),
        },
        'Resistance': {
            'Mean': df['resistance'].mean(),
            'SD': df['resistance'].std(),
            'Min': df['resistance'].min(),
            'Max': df['resistance'].max(),
        }
    }
    
    print(f"\n{'Metric':<15} {'Mean':>10} {'SD':>10} {'Min':>10} {'Max':>10}")
    print("-" * 60)
    for metric, values in stats.items():
        print(f"{metric:<15} {values['Mean']:>9.1%} {values['SD']:>9.1%} "
              f"{values['Min']:>9.1%} {values['Max']:>9.1%}")
    
    # Gap statistics
    gap = df['resistance'] - df['comprehension']
    print(f"\n{'Gap (R-C)':<15} {gap.mean():>9.1%} {gap.std():>9.1%} "
          f"{gap.min():>9.1%} {gap.max():>9.1%}")
    
    # Sample sizes
    print(f"\nTotal observations: {len(df)}")
    print(f"Models: {df['model'].nunique()}")
    print(f"Vulnerability types: {df['cwe'].nunique()}")
    print(f"Instances per model-vuln: {len(df) // (df['model'].nunique() * df['cwe'].nunique())}")

def main():
    print("Loading data...")
    df = load_data()
    
    # Generate tables
    table2 = generate_table2(df)
    table3 = generate_table3(df)
    generate_summary_stats(df)
    
    # Save tables to CSV
    output_dir = Path(__file__).parent.parent / 'data'
    table2.to_csv(output_dir / 'table2_model_comparison.csv', index=False)
    table3.to_csv(output_dir / 'table3_vulnerability_resistance.csv')
    
    print("\n✓ Tables saved to data/ directory")

if __name__ == '__main__':
    main()
