#!/usr/bin/env python3
"""
run_all.py

Master script to execute the complete analysis pipeline.
Reproduces all results from the paper.

Usage:
    python run_all.py           # Run everything
    python run_all.py --quick   # Skip visualizations
    python run_all.py --test    # Just verify scripts work
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse
import time

SCRIPTS = [
    ('00_generate_data.py', 'Generate synthetic experimental data'),
    ('01_data_preprocessing.py', 'Preprocess and validate data'),
    ('02_descriptive_stats.py', 'Generate Tables 2 and 3'),
    ('03_correlation_analysis.py', 'RQ1: Correlation analysis'),
    ('04_statistical_tests.R', 'Statistical validation (R)'),
    ('05_visualizations.py', 'Generate Figures 1-4'),
    ('06_ablation_analysis.py', 'Ablation studies'),
    ('07_thompson_sampling.py', 'Thompson Sampling demo'),
]


def run_script(script_name: str, description: str, skip_r: bool = False) -> bool:
    """Run a single analysis script."""
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"Description: {description}")
    print('='*60)
    
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"⚠ Script not found: {script_path}")
        return False
    
    # Determine interpreter
    if script_name.endswith('.R'):
        if skip_r:
            print("⏭ Skipping R script")
            return True
        cmd = ['Rscript', str(script_path)]
    else:
        cmd = [sys.executable, str(script_path)]
    
    try:
        start = time.time()
        result = subprocess.run(
            cmd,
            cwd=script_path.parent,
            capture_output=False,
            text=True,
            timeout=300  # 5 minute timeout
        )
        elapsed = time.time() - start
        
        if result.returncode == 0:
            print(f"\n✓ Completed in {elapsed:.1f}s")
            return True
        else:
            print(f"\n✗ Failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n✗ Timed out after 300s")
        return False
    except FileNotFoundError as e:
        print(f"\n✗ Interpreter not found: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def check_dependencies():
    """Check that required dependencies are available."""
    print("Checking dependencies...")
    
    issues = []
    
    # Check Python packages
    required_packages = ['pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn']
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} (missing)")
            issues.append(f"pip install {pkg}")
    
    # Check R
    try:
        result = subprocess.run(['Rscript', '--version'], 
                              capture_output=True, timeout=10)
        print(f"  ✓ R/Rscript")
    except:
        print(f"  ⚠ R/Rscript (not found - R scripts will be skipped)")
    
    if issues:
        print(f"\nMissing packages. Install with:")
        print(f"  pip install {' '.join(p.split()[-1] for p in issues)}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Run complete analysis pipeline')
    parser.add_argument('--quick', action='store_true', 
                       help='Skip visualizations')
    parser.add_argument('--test', action='store_true',
                       help='Just verify scripts exist')
    parser.add_argument('--skip-r', action='store_true',
                       help='Skip R scripts')
    parser.add_argument('--from-step', type=int, default=0,
                       help='Start from step N (0-indexed)')
    args = parser.parse_args()
    
    print("="*60)
    print("EASE 2026 REPLICATION PACKAGE")
    print("Complete Analysis Pipeline")
    print("="*60)
    
    # Check dependencies
    if not args.test:
        check_dependencies()
    
    # Filter scripts
    scripts_to_run = SCRIPTS[args.from_step:]
    
    if args.quick:
        scripts_to_run = [(s, d) for s, d in scripts_to_run 
                         if '05_visual' not in s]
    
    if args.test:
        print("\n[TEST MODE] Checking scripts exist...")
        for script, desc in scripts_to_run:
            path = Path(__file__).parent / script
            status = "✓" if path.exists() else "✗"
            print(f"  {status} {script}: {desc}")
        return
    
    # Run scripts
    results = []
    total_start = time.time()
    
    for script, desc in scripts_to_run:
        success = run_script(script, desc, skip_r=args.skip_r)
        results.append((script, success))
    
    total_elapsed = time.time() - total_start
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, s in results if s)
    failed = len(results) - passed
    
    for script, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {script}")
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    print(f"Time: {total_elapsed:.1f}s")
    
    if failed == 0:
        print("\n✓ All analyses completed successfully!")
        print("\nOutputs:")
        print("  - data/combined_results.csv (main dataset)")
        print("  - data/table2_model_comparison.csv")
        print("  - data/table3_vulnerability_resistance.csv")
        print("  - figures/*.png (all figures)")
    else:
        print(f"\n⚠ {failed} script(s) failed. Check output above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
