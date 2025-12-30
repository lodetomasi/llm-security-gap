# Replication Package

## Characterizing the Gap Between Vulnerability Comprehension and Secure Code Generation in LLMs


---

## Overview

This replication package contains all materials necessary to reproduce the experiments and analyses presented in our paper. We provide:

- **704 evaluation prompts** (352 comprehension + 352 generation)
- **Scoring rubric** with detailed criteria and examples
- **Raw experimental data** (all scores, model outputs)
- **Analysis scripts** (Python and R)
- **All 15 prompting strategies** with selection rationale

## Repository Structure

```
replication_package/
├── README.md                    # This file
├── LICENSE.txt                  # MIT License
├── prompts/
│   ├── comprehension/           # 352 comprehension prompts
│   │   ├── sql_injection.json
│   │   ├── command_injection.json
│   │   ├── xss.json
│   │   ├── path_traversal.json
│   │   ├── ssrf.json
│   │   ├── xxe.json
│   │   ├── deserialization.json
│   │   └── code_injection.json
│   └── generation/              # 352 generation prompts (7 strategies × 8 CWEs × ~6 variants)
│       ├── direct_injection.json
│       ├── subtle_obfuscation.json
│       ├── legacy_patterns.json
│       ├── academic_examples.json
│       ├── research_framing.json
│       ├── educational_context.json
│       └── code_review_context.json
├── rubric/
│   ├── scoring_rubric.md        # Complete rubric with criteria
│   ├── scoring_examples.md      # Worked examples for each level
│   └── calibration_samples.json # 20 samples used for rater calibration
├── data/
│   ├── comprehension_scores.csv # Raw comprehension scores (n=352)
│   ├── generation_scores.csv    # Raw generation/resistance scores (n=352)
│   ├── combined_results.csv     # Merged dataset for analysis
│   ├── model_outputs/           # Full model responses (anonymized)
│   └── inter_rater_data.csv     # Data for IRR calculation
├── analysis/
│   ├── requirements.txt         # Python dependencies
│   ├── 01_data_preprocessing.py
│   ├── 02_descriptive_stats.py
│   ├── 03_correlation_analysis.py
│   ├── 04_statistical_tests.R
│   ├── 05_visualizations.py
│   └── 06_ablation_analysis.py
├── strategies/
│   ├── all_15_strategies.md     # Complete list with rationale
│   └── selection_criteria.md    # Why 7 were retained
└── samples/
    ├── vulnerable_code/         # 32 vulnerability samples (4 per CWE)
    └── sample_sources.md        # Sources: CVE, CTF, author-constructed
```

## Quick Start

### 1. Install Dependencies

```bash
# Python dependencies
pip install -r analysis/requirements.txt

# R dependencies (for statistical tests)
Rscript -e "install.packages(c('effsize', 'BayesFactor', 'psych'))"
```

### 2. Reproduce Main Results

```bash
# Generate Table 2 (Comprehension vs Generation by Model)
python analysis/02_descriptive_stats.py

# Generate correlation analysis (ρ=0.21)
python analysis/03_correlation_analysis.py

# Run all statistical tests (Table in Section 4.1)
Rscript analysis/04_statistical_tests.R

# Generate figures
python analysis/05_visualizations.py
```

### 3. Verify Key Findings

```bash
# RQ1: Comprehension-Generation Gap
python -c "
import pandas as pd
df = pd.read_csv('data/combined_results.csv')
print(f'Mean Comprehension: {df.comprehension.mean():.1%}')
print(f'Mean Resistance: {df.resistance.mean():.1%}')
print(f'Spearman rho: {df.comprehension.corr(df.resistance, method=\"spearman\"):.2f}')
"
```

## Experimental Design

### Models Evaluated (n=11)

| Model | Parameters | Type | Access |
|-------|------------|------|--------|
| DeepSeek-Coder | 33B | Code-specialized | Open |
| StarCoder2 | 15B | Code-only | Open |
| CodeLlama | 34B | Code-specialized | Open |
| Qwen-7B | 7B | General + Code | Open |
| Qwen-72B | 72B | General + Code | Open |
| Gemma-7B | 7B | Efficient | Open |
| Mixtral-8x22B | 141B | MoE | Open |
| Llama-3-70B | 70B | Safety-focused | Open |
| GPT-4o | -- | Flagship | Proprietary |
| Claude-3.5-Sonnet | -- | Flagship | Proprietary |
| Gemini-1.5-Pro | -- | Flagship | Proprietary |

### Vulnerability Categories (n=8)

| CWE | Name | Layer |
|-----|------|-------|
| CWE-89 | SQL Injection | Data |
| CWE-78 | Command Injection | System |
| CWE-79 | Cross-Site Scripting (XSS) | Presentation |
| CWE-22 | Path Traversal | File System |
| CWE-918 | Server-Side Request Forgery (SSRF) | Network |
| CWE-611 | XML External Entity (XXE) | Parser |
| CWE-502 | Insecure Deserialization | Object |
| CWE-94 | Code Injection | Execution |

### Sample Sizes

- **352 unique evaluation points**: 11 models × 8 CWEs × 4 instances
- **704 total evaluations**: 352 comprehension + 352 generation
- **8 trials per instance**: For temperature-induced variation
- **32 vulnerability samples**: 4 per CWE category

## Data Dictionary

### combined_results.csv

| Column | Type | Description |
|--------|------|-------------|
| `eval_id` | string | Unique identifier (e.g., "gpt4o_sql_01") |
| `model` | string | Model name |
| `model_type` | string | Category: Code/General/Safety/Proprietary |
| `cwe` | string | CWE identifier (e.g., "CWE-89") |
| `vuln_type` | string | Vulnerability name (e.g., "SQL Injection") |
| `instance_id` | int | Sample instance (1-4) |
| `comprehension` | float | Normalized score 0-1 |
| `comp_identify` | int | Binary: identified vulnerability? |
| `comp_understand` | int | 0-3: explanation quality |
| `comp_fix` | int | 0-3: fix quality |
| `comp_severity` | int | 1-10: severity rating |
| `resistance` | float | Normalized score 0-1 |
| `breach_round` | int | First breach round (1-10) or 0 if never |
| `breach_strategy` | string | Strategy that caused breach (if any) |
| `rater_a` | float | Rater A's comprehension score |
| `rater_b` | float | Rater B's comprehension score |

## Ethical Considerations

- **No human subjects**: Only expert raters (co-investigators)
- **Sandboxed testing**: All vulnerable code tested in isolated Docker containers
- **Responsible disclosure**: No novel attack techniques; all strategies from literature
- **Model access**: Used via official APIs under standard terms of service



## License

This replication package is released under the MIT License. See `LICENSE.txt`.

## Contact

For questions about this replication package, please open an issue in this repository.
