# Changelog

All notable changes to this project will be documented in this file.

This project follows Semantic Versioning.

---

## [1.3.1] - 2026-02-15

### Fixed

### Snippet Copy-Paste Stability

- Resolved string literal breakage caused by embedded `\n` inside triple-quoted snippet templates.
- Replaced vulnerable patterns such as:
  - `print("\n...")`
- Standardized header formatting to:
  - `print()`
  - `print("...")`
- Eliminated potential `SyntaxError: unterminated string literal` when copying snippets directly into external scripts.

---

### Post-Hoc Consistency Adjustments

- Standardized Tukey HSD post-hoc effect size reporting to **Cohen’s d**.
- Ensured Games–Howell post-hoc reporting consistently uses **Hedges’ g**.
- Harmonized effect size column labels and printed output for clarity and consistency.

---

### Execution Safety Validation

- Performed full in-memory compilation check of all snippet templates.
- Verified successful execution under `--smoketest` mode.
- Confirmed:
  - No syntax errors
  - No argparse conflicts
  - No runtime exceptions
  - No snippet compilation failures
- Validated compatibility with current SciPy and statsmodels environments.

---

### Stability Status

- All 36 snippet templates compile without syntax errors.
- CLI argument handling verified.
- No interactive execution crashes detected.
- System validated as Release Candidate–ready.

---

## [1.3.0] - 2026-02-15

### Refactored

### ANOVA Result Output Simplification

- Removed raw `print(res)` output from `anova_oneway()` results.
- Suppressed verbose internal attributes including:
  - `means`
  - `vars_`
  - `nobs`
  - `tuple`
  - auxiliary statistic containers
- Standardized output to display only:
  - F statistic
  - numerator and denominator degrees of freedom
  - p-value
- Implemented formatted reporting via `f-string`:
  - `F(df1, df2) = value, p = value`
- Added conditional interpretation block based on alpha level.

---

### Added

### Post-Hoc Effect Size for Statistically Significant Pairs (3+ Groups)

- Implemented pairwise effect size computation limited to statistically significant comparisons.
- Tukey HSD:
  - Computes **Cohen’s d** only when `reject == True`.
- Games–Howell:
  - Computes or extracts **Hedges’ g** when `pval < alpha`.
- Kruskal–Wallis workflow:
  - Maintains global η²_H reporting.
  - No mean-based pairwise effect size added.
- No expansion into additional effect size families or CI-based reporting.

---

### Notes

- Improves output readability and interpretability.
- Does not modify statistical logic.
- No changes to routing, assumptions, or model selection logic.
- Maintains backward compatibility with existing workflows.

---

## [1.2.0] - 2026-02-15

### Fixed

- Replaced the “Effect size r (approx) from z” block in the English Wilcoxon one-sample snippet.
- Replaced the paired Wilcoxon z-based effect size block in the Korean file.
- Replaced the Korean one-sample Wilcoxon z-based effect size block.
- Resolved `GofChisquarePower.power()` missing `n_bins` argument error in chi-square goodness-of-fit power analysis.
- Ensured correct `n_bins` handling for valid chi-square power computation.
- Verified all snippet keys and routing references to prevent missing or mismatched calls.
- Confirmed all modules compile without syntax errors and execute without runtime exceptions.
- Removed Bartlett test and standardized equal-variance checking to Levene’s test.
- Eliminated `KeyError: (0, 0)` from residual indexing in chi-square independence workflows.
- Removed Pylance operator-type warnings related to:
  - `expected < 1`
  - `chi2 / (n * k)`
  - tuple / Series arithmetic conflicts.
- Eliminated unsafe tuple unpacking pattern from `chi2_contingency`.
- Replaced all ambiguous return unpacking with object-based access:
  - `res = stats.chi2_contingency(...)`
  - `res.expected_freq`
- Removed all raw exception detail printing (`print(e)` patterns).
- Prevented full traceback exposure in runtime execution blocks.
- Ensured no remaining `ZeroDivisionError` or shape-related runtime crashes in chi-square workflows.

---

### Refactored

### Chi-Square Workflows (Full Stabilization)

- Converted all contingency tables to NumPy arrays via:
  - `table_np = np.asarray(table)`
- Replaced `.sum()` operations that could return pandas `Series` with guaranteed scalar:
  - `float(table_np.sum())`
- Standardized Cramer's V computation:
  - Safe handling when `n == 0`
  - Safe handling when `min(r-1, c-1) == 0`
  - Returns `np.nan` when undefined instead of raising errors.
- Replaced `statsmodels.Table.standardized_resids` usage with manual adjusted standardized residual computation.
- Standardized residual logic across:
  - Main independence test
  - Category collapse alternative
  - Monte Carlo fallback
- Unified ndarray-safe pattern across all chi-square branches.

---

### Added

- Added `CHANGELOG.md` file to formally track project changes.

### Fisher’s Exact Test (2×2) Enhancements

- Added **Phi coefficient (φ)** effect size computation for 2×2 Fisher’s Exact Test.
  - Implemented as:
    - `φ = sqrt(χ² / n)` (chi-square equivalent formulation)
  - Safe handling when `n == 0`
  - Returns `np.nan` when undefined.
- Integrated φ into centralized `EFFECT_GUIDE` interpretation system.
- Effect size now automatically displayed alongside:
  - Odds Ratio
  - p-value
- Maintained stable execution without altering Fisher test core logic.

### Fisher–Freeman–Halton Test Enhancements (R×C Exact Test)

- Added **Cramer's V** effect size computation.
  - ndarray-normalized computation
  - Safe denominator handling
  - Returns `np.nan` when undefined.
- Implemented post-hoc residual analysis:
  - Adjusted standardized residuals
  - `|Residual| > 2` flagged as meaningful deviation.
- Unified residual computation logic with chi-square independence workflow.
- Ensured no dependency on `statsmodels.Table.standardized_resids`.
- Maintained ndarray-only arithmetic for full static-type compliance.

### Expanded Post-Hoc Support (Exact Tests)

- Enabled residual-based diagnostics for:
  - Fisher’s Exact Test (2×2)
  - Fisher–Freeman–Halton (R×C)
- Standardized output structure:
  - Main test result
  - Effect size
  - Residual matrix (when applicable)
  - Interpretation guidance
- Post-hoc logic unified with chi-square independence residual engine.

---

### Hardened Runtime Stability & Static-Type Compliance

### Global ndarray Normalization

- Enforced explicit casting for:
  - `table`
  - `observed`
  - `expected`
  - residual matrices
- Eliminated unintended pandas object propagation.
- Prevented Series arithmetic inside scalar effect size formulas.

### Cochran’s Rule Improvements

- Cast expected counts to NumPy arrays before comparison.
- Eliminated operator-type mismatch warnings for:
  - `expected < 1`
  - `expected < 5`
- Standardized `shape` output via `table_np.shape`.

### Monte Carlo & Collapse Stability

- Prevented zero-total division in simulated probability vectors.
- Ensured row/column sum calculations operate strictly on ndarray.
- Unified collapse and Monte Carlo branches under identical safe pattern.

### Effect Size Stability

Verified safe arithmetic across:

- Cohen’s d  
- Hedges’ g  
- Cohen’s h  
- Pearson’s r  
- Spearman’s ρ  
- Kendall’s W  
- η²  
- Cohen’s f  
- ε²  
- Cramer's V  
- Cohen’s w  
- Phi coefficient (φ)  
- Cliff’s delta  
- Odds Ratio  

- Ensured all denominators are float scalars.
- Removed all tuple / Series operator mismatches.

---

### Hypothesis Direction & Confidence Interval Improvements

- Global synchronization of alternative hypothesis selection (`two-sided`, `greater`, `less`) across all applicable tests.
- One-sided confidence interval logic:
  - `greater` → CI = (lower_bound, +∞)
  - `less` → CI = (-∞, upper_bound)
- Bootstrap confidence intervals updated to respect one-sided hypothesis when applicable.
- Alternative hypothesis prompt now appears only when statistically meaningful.
- Session-level caching of alternative hypothesis selection.

---

### Expanded Effect Size Support

- Introduced centralized `EFFECT_GUIDE` dictionary.
- Standardized interpretation guidance for:

  - Cohen’s d  
  - Hedges’ g  
  - Cohen’s h  
  - Pearson’s r  
  - Spearman’s ρ  
  - Kendall’s W  
  - η²  
  - Cohen’s f  
  - ε²  
  - Cramer's V  
  - Cohen’s w  
  - Phi coefficient (φ)  
  - Cliff’s delta  
  - Odds Ratio  

- Effect size interpretation automatically displayed with each relevant test.

---

### Chi-Square Enhancements

- Integrated automatic Cochran’s Rule checking.
- Implemented fallback logic:
  - 2×2 tables → Fisher’s Exact Test
  - R×C tables → Category collapsing, Fisher–Freeman–Halton (if supported), or Monte Carlo simulation.
- Added post-hoc residual analysis:
  - Standardized residuals (GOF)
  - Adjusted standardized residuals (Independence & Exact tests)
  - `|Residual| > 2` flagged as meaningful deviation.
- Separated Cochran check from main test generation.
- Simplified GOF and independence snippets to focus on:
  - Main test
  - Effect size
  - Residual diagnostics

---

### Cognitive Flow Improvements (Count Data)

- Simplified count-data routing.
- After selecting “Count” as dependent variable, users directly choose among:
  1. Distribution vs expected distribution → Chi-square Goodness-of-Fit
  2. Relationship between two categorical variables → Chi-square Independence
  3. Event counts relative to exposure/time → Poisson test or rate comparison
- Eliminated redundant intermediate branching.
- Preserved statistical separation between GOF, independence, and Poisson models.

---

### Comprehensive TODO Normalization & Input Contract Definition

- Performed full audit of all snippet templates.
- Standardized `# TODO` placement across Korean and English versions.
- Introduced explicit `# TODO INPUT:` blocks.
- Defined clear input contracts for:
  - One-sample tests
  - Two-sample tests
  - Paired tests
  - Multi-group tests
  - Proportion tests
  - Chi-square tests
  - Regression models
  - Logistic regression (binary constraint clarified)
  - Bootstrap and Monte Carlo routines
- Added unified INPUT CHECKLIST block clarifying:
  - dtype expectations
  - shape requirements
  - non-negativity constraints
  - equal-length requirements
  - logistic 0/1 constraint
- Reduced ambiguity and improved reproducibility.

---

### Execution Safety Validation

- Performed full routing verification.
- Confirmed snippet key integrity.
- Verified compatibility with current SciPy and statsmodels APIs.
- Eliminated tuple-type inference ambiguities.
- Ensured stable execution across all chi-square branches.
- Confirmed no remaining:

  - `KeyError`
  - `ZeroDivisionError`
  - Operator "/" not supported
  - Operator "<" not supported

---

## [1.1.0] - 2026-02-14

### Fixed
- Removed Bartlett test for equal variance checking.
- Standardized equal variance testing to use Levene’s test (KR/EN support).

---

## [1.0.0] - 2026-02-14

### Added
- Initial project structure
- Core statistical navigation functionality
- README documentation
- License
- Requirements file
- Execution script (`run.py`)
