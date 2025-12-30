#!/usr/bin/env Rscript
# 04_statistical_tests.R
#
# Comprehensive statistical tests for RQ1 analysis.
# Reproduces statistical validation from Section 4.1 of the paper.

# Install packages if needed
packages <- c("effsize", "BayesFactor", "psych", "boot")
for (pkg in packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
    library(pkg, character.only = TRUE)
  }
}

cat("=" , rep("=", 59), "\n", sep="")
cat("STATISTICAL TESTS FOR RQ1\n")
cat("=" , rep("=", 59), "\n\n", sep="")

# Load data
data_path <- file.path(dirname(sys.frame(1)$ofile), "..", "data", "combined_results.csv")
if (!file.exists(data_path)) {
  data_path <- "../data/combined_results.csv"
}
df <- read.csv(data_path)

cat("Data loaded:", nrow(df), "observations\n\n")

# ============================================================
# 1. PAIRED T-TEST
# ============================================================
cat("-" , rep("-", 39), "\n", sep="")
cat("1. PAIRED T-TEST\n")
cat("-" , rep("-", 39), "\n\n", sep="")

t_result <- t.test(df$resistance, df$comprehension, paired = TRUE)
print(t_result)

cat("\nDegrees of freedom:", t_result$parameter, "\n")
cat("t-statistic:", round(t_result$statistic, 2), "\n")
cat("p-value:", format(t_result$p.value, scientific = TRUE), "\n")

# ============================================================
# 2. EFFECT SIZE (Cohen's d)
# ============================================================
cat("\n", "-" , rep("-", 39), "\n", sep="")
cat("2. EFFECT SIZE (Cohen's d)\n")
cat("-" , rep("-", 39), "\n\n", sep="")

d_result <- cohen.d(df$resistance, df$comprehension, paired = TRUE)
print(d_result)

cat("\nCohen's d:", round(d_result$estimate, 2), "\n")
cat("95% CI: [", round(d_result$conf.int[1], 2), ", ", 
    round(d_result$conf.int[2], 2), "]\n", sep="")
cat("Interpretation:", d_result$magnitude, "\n")

# ============================================================
# 3. WILCOXON SIGNED-RANK TEST (Non-parametric)
# ============================================================
cat("\n", "-" , rep("-", 39), "\n", sep="")
cat("3. WILCOXON SIGNED-RANK TEST\n")
cat("-" , rep("-", 39), "\n\n", sep="")

w_result <- wilcox.test(df$resistance, df$comprehension, paired = TRUE)
print(w_result)

# Effect size r = Z / sqrt(N)
z_score <- qnorm(w_result$p.value / 2)
r_effect <- abs(z_score) / sqrt(nrow(df))
cat("\nEffect size r:", round(r_effect, 2), "\n")

# ============================================================
# 4. BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================
cat("\n", "-" , rep("-", 39), "\n", sep="")
cat("4. BOOTSTRAP CONFIDENCE INTERVAL\n")
cat("-" , rep("-", 39), "\n\n", sep="")

mean_diff <- function(data, indices) {
  d <- data[indices, ]
  return(mean(d$resistance - d$comprehension))
}

set.seed(42)
boot_result <- boot(df, mean_diff, R = 10000)
boot_ci <- boot.ci(boot_result, type = "perc")

cat("Mean difference:", round(boot_result$t0, 4), "\n")
cat("Bootstrap SE:", round(sd(boot_result$t), 4), "\n")
cat("95% CI (percentile): [", round(boot_ci$percent[4], 4), ", ", 
    round(boot_ci$percent[5], 4), "]\n", sep="")

# ============================================================
# 5. BAYESIAN ANALYSIS
# ============================================================
cat("\n", "-" , rep("-", 39), "\n", sep="")
cat("5. BAYESIAN ANALYSIS\n")
cat("-" , rep("-", 39), "\n\n", sep="")

bf_result <- ttestBF(df$resistance, df$comprehension, paired = TRUE)
print(bf_result)

bf_value <- extractBF(bf_result)$bf
cat("\nBayes Factor (BF10):", format(bf_value, scientific = TRUE), "\n")
cat("Interpretation: ", 
    ifelse(bf_value > 100, "Decisive evidence for H1",
    ifelse(bf_value > 30, "Very strong evidence for H1",
    ifelse(bf_value > 10, "Strong evidence for H1",
    ifelse(bf_value > 3, "Moderate evidence for H1",
    "Weak evidence")))), "\n", sep="")

# ============================================================
# 6. CORRELATION ANALYSIS
# ============================================================
cat("\n", "-" , rep("-", 39), "\n", sep="")
cat("6. SPEARMAN CORRELATION\n")
cat("-" , rep("-", 39), "\n\n", sep="")

cor_result <- cor.test(df$comprehension, df$resistance, method = "spearman")
print(cor_result)

cat("\nSpearman rho:", round(cor_result$estimate, 3), "\n")
cat("rho-squared (variance explained):", round(cor_result$estimate^2, 3), "\n")

# ============================================================
# 7. INTER-RATER RELIABILITY
# ============================================================
cat("\n", "-" , rep("-", 39), "\n", sep="")
cat("7. INTER-RATER RELIABILITY\n")
cat("-" , rep("-", 39), "\n\n", sep="")

# Cohen's Kappa for agreement
rater_matrix <- cbind(df$rater_a, df$rater_b)
icc_result <- ICC(rater_matrix)

cat("ICC Results:\n")
print(icc_result)

# ============================================================
# SUMMARY TABLE (matching paper Section 4.1)
# ============================================================
cat("\n", "=" , rep("=", 59), "\n", sep="")
cat("SUMMARY TABLE (Paper Section 4.1)\n")
cat("=" , rep("=", 59), "\n\n", sep="")

summary_table <- data.frame(
  Test = c("Paired t-test", "Cohen's d", "Wilcoxon", "Bayes Factor"),
  Statistic = c(
    paste0("t(", t_result$parameter, ")=", round(t_result$statistic, 2)),
    round(d_result$estimate, 2),
    paste0("W=", round(w_result$statistic, 0)),
    format(bf_value, scientific = TRUE, digits = 2)
  ),
  P_value = c(
    format(t_result$p.value, scientific = TRUE, digits = 2),
    "-",
    format(w_result$p.value, scientific = TRUE, digits = 2),
    "-"
  ),
  CI_95 = c(
    paste0("[", round(boot_ci$percent[4]*100, 1), "%, ", 
           round(boot_ci$percent[5]*100, 1), "%]"),
    paste0("[", round(d_result$conf.int[1], 2), ", ", 
           round(d_result$conf.int[2], 2), "]"),
    "-",
    "-"
  )
)

print(summary_table, row.names = FALSE)

cat("\n✓ Statistical tests complete\n")
