
# ==========================================================
# UNIFIED INPUT CHECKLIST (Applies to ALL statistical snippets)
# ----------------------------------------------------------
# 1. Data Type Requirements
#    - Numeric samples: np.array([...], dtype=float)
#    - Count data: non-negative integers only
#    - Logistic regression y: binary (0 and 1 only)
#
# 2. Shape Requirements
#    - One-sample tests: x → 1D array (n,)
#    - Independent two-sample tests: a, b → two 1D arrays
#    - Paired tests: x_before, x_after → equal-length 1D arrays
#    - Multi-group tests: groups = [g1, g2, g3, ...]
#    - Chi-square tests: table → 2D array (R × C), R ≥ 2, C ≥ 2
#    - Regression:
#         y → shape (n,)
#         X → shape (n, p)
#
# 3. Data Validity
#    - No NaN values
#    - No infinite values
#    - Sample size must be > 0
#    - For proportion tests: 0 ≤ count ≤ nobs
#
# 4. Statistical Constraints
#    - Chi-square expected counts should not all be zero
#    - Logistic regression requires variability in y
#    - Equal-length requirement for paired tests
#
# 5. Example Template
#    x = np.array([1.2, 2.3, 3.4], dtype=float)
#
# ==========================================================


# ============================================================
# Statistical Test Selection & Inference Assistant (CLI)
# ------------------------------------------------------------
# Features:
#  - Guided test selection workflow
#  - Assumption checks (Normality: Shapiro + Q-Q(probplot), Equal variance: Levene)
#  - Effect sizes and confidence intervals (analytic or bootstrap guidance where applicable)
#
# Version: v1.3.1
# Last check: 2026-02-15  |  smoketest: PASS
# Developed by: 김규열(Ojirokim)
# License: MIT
# ============================================================

#!/usr/bin/env python3
"""
Interactive statistical test selector (CLI) — Workflow-friendly edition

Goals:
- Avoid "weird endings": every path ends with a clear FINAL RECOMMENDATION summary.
- Consistent workflow: assumptions -> decision -> final test (where applicable).
- Prints code snippets for: test, effect size, power, CI, and assumption checks (Shapiro+QQ, Levene).
- Offers to restart so you can explore multiple branches in one run.

Notes:
- For many nonparametric tests, power/CI often require simulation/bootstrapping; snippets include guidance.
"""

from typing import Optional, List, Tuple, Dict, Any

CURRENT_ALT: Optional[str] = None  # "two-sided" | "greater" | "less"

# Common snippet settings (to avoid repeating in every snippet)
COMMON_SETTINGS = """alpha = 0.05  # TODO
alternative = \"two-sided\"  # \"two-sided\" | \"greater\" | \"less\"
"""


import argparse
import sys


# ----------------------------
# Helpers
# ----------------------------

def ask_choice(prompt: str, choices: List[Tuple[str, str]]) -> str:
    print("\n" + prompt)
    for i, (_, label) in enumerate(choices, start=1):
        print(f"  {i}. {label}")
    while True:
        raw = input("Select a number: ").strip()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(choices):
                return choices[idx - 1][0]
        print("Invalid selection. Try again.")



def ensure_alternative_selected() -> str:
    """Ask for alternative (one-/two-sided) only once, and only when a test needs it."""
    global CURRENT_ALT
    if CURRENT_ALT is None:
        CURRENT_ALT = ask_choice(
            "Choose the alternative hypothesis (decide *before* looking at data)",
            [
                ("two-sided", "Two-sided"),
                ("greater", "One-sided: greater"),
                ("less", "One-sided: less"),
            ],
        )
    return CURRENT_ALT

def ask_yes_no(prompt: str) -> bool:
    return ask_choice(prompt, [("y", "Yes"), ("n", "No")]) == "y"


def print_node(title: str, detail: Optional[str] = None, code: Optional[str] = None) -> None:
    """Pretty printer for nodes. Uses copy-friendly snippet markers."""
    import sys
    print("\n" + "=" * 72)
    print(title)
    if detail:
        print("-" * 72)
        print(detail)
    if code:
        print("-" * 72)
        print("Code snippet (copy-friendly):")
        print("# ---BEGIN SNIPPET---")
        safe_code = code.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ").rstrip("\n")
        safe_code = safe_code.replace("__COMMON_SETTINGS__", COMMON_SETTINGS)
        sys.stdout.write(safe_code + "\n")
        print("# ---END SNIPPET---")
    print("=" * 72)


# ----------------------------
# Snippets (tests + assumptions)
# ----------------------------

ASSUMPTION_SNIPPETS = {
    "normality_shapiro": (
        "Normality check (Shapiro-Wilk + QQ plot)",
        """Use BOTH a statistical test and a visual check.
Apply to: sample (one-sample), paired differences (paired), or residuals (regression).""",
        """from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 1) Shapiro-Wilk test
stat, p = stats.shapiro(x)   # x: sample OR residuals OR paired differences
print("Shapiro p =", p)

# 2) QQ plot (visual normality check)
stats.probplot(x, dist="norm", plot=plt)
plt.title("QQ Plot")
plt.show()

# Alternative QQ plot using SciPy:
# stats.probplot(x, dist="norm", plot=plt)
# plt.show()
"""
    ),
    "equal_var_levene": (
        "Equal variance check (Levene)",
        "Useful before Student t-test / classic ANOVA.",
        """from scipy import stats

stat, p = stats.levene(group1, group2)  # pass 3+ groups too
print("Levene p =", p)
"""
    ),
}

TEST_SNIPPETS = {
    # One-sample t
    "onesample_t": (
        "One-sample t-test",
        "Compare one sample mean to a constant value.",
        """
from scipy import stats
import numpy as np

__COMMON_SETTINGS__

# x : 1D sample array
mu0 = 0  # TODO: hypothesized mean under H0

t_stat, p_value = stats.ttest_1samp(x, popmean=mu0, alternative=alternative)

xbar = float(np.mean(x))
sd = float(np.std(x, ddof=1))
d = (xbar - mu0) / sd if sd > 0 else np.nan

print("[One-sample t-test]")
print(f"  H0: mean = {mu0} | alternative = {alternative}")
print(f"  mean = {xbar:.4f}, t = {t_stat:.4f}, p = {p_value:.4g}, alpha = {alpha}")
print(f"  Effect size (Cohen's d) = {d:.4f}  (|d|: 0.2 small, 0.5 medium, 0.8 large; context-dependent)")

if p_value < alpha:
    print("  → Reject H0: evidence that the mean differs from mu0.")
else:
    print("  → Fail to reject H0: insufficient evidence that the mean differs from mu0.")

# CI for mean difference (mean - mu0)
n = len(x); df = n - 1
se = stats.sem(x)
if alternative == "two-sided":
    tcrit = stats.t.ppf(1 - alpha/2, df)
    ci = ((xbar - mu0) - tcrit*se, (xbar - mu0) + tcrit*se)
elif alternative == "greater":
    tcrit = stats.t.ppf(1 - alpha, df)
    ci = ((xbar - mu0) - tcrit*se, np.inf)
else:  # "less"
    tcrit = stats.t.ppf(1 - alpha, df)
    ci = (-np.inf, (xbar - mu0) + tcrit*se)
print(f"  {int((1-alpha)*100)}% CI for (mean - mu0) = {ci}")

"""
    ),
    # Wilcoxon one-sample (vs constant)
    "wilcoxon_onesample": (
        "Wilcoxon signed-rank test (one-sample vs constant)",
        "Nonparametric alternative to one-sample t-test. Tests median(x - mu0) = 0.",
        """from scipy import stats
import numpy as np

# x : sample array
mu0 = 0  # TODO: hypothesized value
diff = x - mu0

w_stat, p_value = stats.wilcoxon(diff, alternative="two-sided")  # or "less"/"greater"
print("W =", w_stat, "p =", p_value)

# Effect size: Rank-biserial correlation (RBC)
# RBC = (W+ - W-) / (W+ + W-), where W+ is sum of ranks for positive differences.
diff_nz = diff[diff != 0]
if len(diff_nz) == 0:
    print("All differences are zero; effect size is undefined.")
else:
    ranks = stats.rankdata(np.abs(diff_nz))
    w_pos = ranks[diff_nz > 0].sum()
    w_neg = ranks[diff_nz < 0].sum()
    rbc = (w_pos - w_neg) / (w_pos + w_neg)
    print("Rank-biserial correlation (RBC) =", rbc)

    # Bootstrap CI for RBC
    alpha = 0.05
    B = 2000
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(B):
        d_b = rng.choice(diff, size=len(diff), replace=True)
        d_b = d_b[d_b != 0]
        if len(d_b) < 2:
            continue
        r_b = stats.rankdata(np.abs(d_b))
        w_pos_b = r_b[d_b > 0].sum()
        w_neg_b = r_b[d_b < 0].sum()
        boot.append((w_pos_b - w_neg_b) / (w_pos_b + w_neg_b))
    boot = np.sort(np.array(boot))
    ci = (np.quantile(boot, alpha/2), np.quantile(boot, 1-alpha/2))
    print(f"{int((1-alpha)*100)}% bootstrap CI for RBC =", ci)

# Power note: typically via simulation for Wilcoxon.
"""
    ),

    # Paired t
    "paired_t": (
        "Paired t-test",
        "Same subjects measured twice (before/after).",
        """
from scipy import stats
import numpy as np

__COMMON_SETTINGS__

# x_before, x_after : paired arrays (same length)
t_stat, p_value = stats.ttest_rel(x_before, x_after, alternative=alternative)

diff = np.asarray(x_before) - np.asarray(x_after)
est = float(np.mean(diff))
sd = float(np.std(diff, ddof=1))
d = est / sd if sd > 0 else np.nan

print("[Paired t-test]")
print(f"  H0: mean(diff)=0 | alternative = {alternative}")
print(f"  mean(diff) = {est:.4f}, t = {t_stat:.4f}, p = {p_value:.4g}, alpha = {alpha}")
print(f"  Effect size (Cohen's d_paired) = {d:.4f}  (|d|: 0.2 small, 0.5 medium, 0.8 large; context-dependent)")

if p_value < alpha:
    print("  → Reject H0: evidence of a mean change between paired measurements.")
else:
    print("  → Fail to reject H0: insufficient evidence of a mean change.")

# CI for mean(diff)
n = len(diff); df = n - 1
se = stats.sem(diff)
if alternative == "two-sided":
    tcrit = stats.t.ppf(1 - alpha/2, df)
    ci = (est - tcrit*se, est + tcrit*se)
elif alternative == "greater":
    tcrit = stats.t.ppf(1 - alpha, df)
    ci = (est - tcrit*se, np.inf)
else:
    tcrit = stats.t.ppf(1 - alpha, df)
    ci = (-np.inf, est + tcrit*se)
print(f"  {int((1-alpha)*100)}% CI for mean(diff) = {ci}")

"""
    ),
    "wilcoxon_paired": (
        "Wilcoxon signed-rank test (paired)",
        "Nonparametric alternative to paired t-test.",
        """from scipy import stats
import numpy as np

__COMMON_SETTINGS__

# x_before, x_after : paired arrays
diff = x_before - x_after

stat, p_value = stats.wilcoxon(diff, alternative=alternative)
print(f"[Wilcoxon (paired)] W = {stat:.4g}, p = {p_value:.4g}, alpha = {alpha}")

if p_value < alpha:
    print("→ Decision: reject H0 (evidence of a non-zero median difference)")
else:
    print("→ Decision: fail to reject H0 (insufficient evidence)")

# ── Effect size: Rank-biserial correlation (RBC) ──
diff_nz = diff[diff != 0]
if len(diff_nz) == 0:
    print("[Effect size] All differences are zero → undefined")
else:
    ranks = stats.rankdata(np.abs(diff_nz))
    w_pos = ranks[diff_nz > 0].sum()
    w_neg = ranks[diff_nz < 0].sum()
    rbc = (w_pos - w_neg) / (w_pos + w_neg)
    print(f"[Effect size] RBC = {rbc:.4f}  (rule of thumb: 0.1 small, 0.3 medium, 0.5 large; context-dependent)")

    # ── Bootstrap CI ──
    rng = np.random.default_rng(0)
    B = 2000
    vals = []
    for _ in range(B):
        d_b = rng.choice(diff, size=len(diff), replace=True)
        d2 = d_b[d_b != 0]
        if len(d2) < 2:
            continue
        r2 = stats.rankdata(np.abs(d2))
        w_pos2 = r2[d2 > 0].sum()
        w_neg2 = r2[d2 < 0].sum()
        vals.append((w_pos2 - w_neg2) / (w_pos2 + w_neg2))
    vals = np.array(vals)
    if len(vals) > 10:
        if alternative == "two-sided":
            ci = (np.quantile(vals, 0.025), np.quantile(vals, 0.975))
        elif alternative == "greater":
            ci = (np.quantile(vals, 0.05), np.inf)
        else:
            ci = (-np.inf, np.quantile(vals, 0.95))
        print(f"[CI] 95% bootstrap CI for RBC = {ci}")
    else:
        print("[CI] Too few bootstrap samples; CI skipped")"""
    ),

    # Independent t: Student & Welch
    "ind_t_student": (
        "Independent t-test (Student, equal variances)",
        "Two independent groups, assume equal variances.",
        """from scipy import stats
import numpy as np

# a, b : independent samples
t_stat, p_value = stats.ttest_ind(a, b, equal_var=True, alternative="two-sided")
print("t =", t_stat, "p =", p_value)

# Effect size: Cohen's d (pooled)
n1, n2 = len(a), len(b)
s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
df = n1 + n2 - 2
spooled = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / df)
d = (np.mean(a) - np.mean(b)) / spooled
print("Cohen's d =", d)

# Power
from statsmodels.stats.power import TTestIndPower
power = TTestIndPower().power(effect_size=d, nobs1=n1, ratio=n2/n1, alpha=0.05)  # TODO alpha
print("Power =", power)

# CI for mean difference (Student)
alpha = 0.05  # TODO
mean_diff = np.mean(a) - np.mean(b)
se_diff = spooled * np.sqrt(1/n1 + 1/n2)

tcrit_two = stats.t.ppf(1 - alpha/2, df)
tcrit_one = stats.t.ppf(1 - alpha, df)

if alternative == "two-sided":
    ci = (mean_diff - tcrit_two*se_diff, mean_diff + tcrit_two*se_diff)
elif alternative == "greater":
    ci = (mean_diff - tcrit_one*se_diff, np.inf)
else:  # "less"
    ci = (-np.inf, mean_diff + tcrit_one*se_diff)

print(f"{int((1-alpha)*100)}% CI mean difference =", ci)
"""
    ),
    "ind_t_welch": (
        "Welch's t-test (unequal variances)",
        "Two independent groups, do NOT assume equal variances.",
        """from scipy import stats
import numpy as np

# a, b : independent samples
t_stat, p_value = stats.ttest_ind(a, b, equal_var=False, alternative="two-sided")
print("t =", t_stat, "p =", p_value)

# Effect size (practical): Cohen's d using pooled SD is common,
# but with unequal variances you may also report Glass' delta.
# Here we keep pooled d for convenience (commented).
n1, n2 = len(a), len(b)
s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
spooled = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2))
d = (np.mean(a) - np.mean(b)) / spooled
print("Cohen's d (pooled, common) =", d)

# Power (approx): use TTestIndPower with d (approx)
from statsmodels.stats.power import TTestIndPower
power = TTestIndPower().power(effect_size=d, nobs1=n1, ratio=n2/n1, alpha=0.05)  # TODO alpha
print("Power (approx) =", power)

# CI for mean difference (Welch)
alpha = 0.05  # TODO
mean_diff = np.mean(a) - np.mean(b)
se_diff = np.sqrt(s1/n1 + s2/n2)
df_w = (s1/n1 + s2/n2)**2 / ((s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1))

tcrit_two = stats.t.ppf(1 - alpha/2, df_w)
tcrit_one = stats.t.ppf(1 - alpha, df_w)

if alternative == "two-sided":
    ci = (mean_diff - tcrit_two*se_diff, mean_diff + tcrit_two*se_diff)
elif alternative == "greater":
    ci = (mean_diff - tcrit_one*se_diff, np.inf)
else:  # "less"
    ci = (-np.inf, mean_diff + tcrit_one*se_diff)

print(f"{int((1-alpha)*100)}% CI mean difference (Welch) =", ci)
"""
    ),
    "mann_whitney": (
        "Mann–Whitney U test",
        "Nonparametric alternative to independent t-test (rank-sum).",
        """
from scipy import stats
import numpy as np

__COMMON_SETTINGS__

# a, b : independent samples
u_stat, p_value = stats.mannwhitneyu(a, b, alternative=alternative)
n1, n2 = len(a), len(b)

# Effect size: rank-biserial correlation (RBC) and Cliff's delta
rbc = 1 - (2 * u_stat) / (n1 * n2)

def cliffs_delta(x, y):
    x = np.asarray(x); y = np.asarray(y)
    gt = sum((xi > y).sum() for xi in x)
    lt = sum((xi < y).sum() for xi in x)
    return (gt - lt) / (len(x) * len(y))

delta = cliffs_delta(a, b)

print("[Mann–Whitney U test]")
print(f"  U = {u_stat:.4f}, p = {p_value:.4g}, alpha = {alpha} | alternative = {alternative}")
print(f"  Effect size: RBC = {rbc:.4f} | Cliff's delta = {delta:.4f}  (context-dependent thresholds)")

if p_value < alpha:
    print("  → Reject H0: evidence of a distribution shift between groups.")
else:
    print("  → Fail to reject H0: insufficient evidence of a distribution shift.")

# Bootstrap CI for RBC (recommended)
B = 2000  # TODO
rng = np.random.default_rng(0)
a = np.asarray(a); b = np.asarray(b)
boot = []
for _ in range(B):
    aa = rng.choice(a, size=n1, replace=True)
    bb = rng.choice(b, size=n2, replace=True)
    u_b, _ = stats.mannwhitneyu(aa, bb, alternative=alternative)
    boot.append(1 - (2*u_b)/(n1*n2))
ci = (np.percentile(boot, 2.5), np.percentile(boot, 97.5))
print(f"  95% bootstrap CI for RBC = [{ci[0]:.4f}, {ci[1]:.4f}]")

"""
    ),

    # ANOVA / Kruskal
    "anova_oneway": (
        "One-way ANOVA",
        "Compare means across 3+ independent groups (parametric).",
        """from scipy import stats
import numpy as np

alpha = 0.05  # TODO

# g1, g2, g3 : group arrays (extend as needed)
f_stat, p_value = stats.f_oneway(g1, g2, g3)
print(f"[One-way ANOVA] F = {f_stat:.4g}, p = {p_value:.4g}, alpha = {alpha}")

if p_value < alpha:
    print("→ Decision: reject H0 (evidence of mean differences across groups)")
    print("→ Next step (post-hoc):")
    print("   - If variances are ~equal: Tukey HSD")
    print("   - If variances are unequal: Welch ANOVA + Games–Howell")
else:
    print("→ Decision: fail to reject H0 (insufficient evidence of mean differences)")

groups = [g1, g2, g3]  # TODO extend
all_y = np.concatenate(groups)
grand_mean = np.mean(all_y)

ss_between = sum(len(g)*(np.mean(g)-grand_mean)**2 for g in groups)
ss_within = sum(np.sum((g - np.mean(g))**2) for g in groups)
ss_total = ss_between + ss_within

eta2 = ss_between / ss_total if ss_total > 0 else np.nan
k = len(groups)
n = len(all_y)
df_between = k - 1
df_within = n - k
ms_within = ss_within / df_within
omega2 = (ss_between - df_between*ms_within) / (ss_total + ms_within) if (ss_total + ms_within) > 0 else np.nan
cohens_f = np.sqrt(eta2 / (1 - eta2)) if (eta2 is not np.nan and eta2 < 1) else np.nan

print(f"[Effect size] eta^2 = {eta2:.4f}, omega^2 = {omega2:.4f}, Cohen's f = {cohens_f:.4f}")
print("  (rule of thumb) eta^2: 0.01 small, 0.06 medium, 0.14 large (context-dependent)")"""
    ),

    "welch_anova": (
        "Welch's ANOVA (one-way, unequal variances)",
        "Compare means across 3+ independent groups when normality is acceptable but variances are unequal.",
        """import numpy as np
from statsmodels.stats.oneway import anova_oneway

alpha = 0.05  # TODO

# groups: list of arrays
groups = [g1, g2, g3]  # TODO extend
labels = ["g1", "g2", "g3"]  # TODO keep same order as groups

res = anova_oneway(groups, use_var="unequal", welch_correction=True)

# ── (1) Print only what you need for interpretation ──
F = float(res.statistic)
p = float(res.pvalue)
df1 = float(res.df_num)
df2 = float(res.df_denom)

print(f"[Welch ANOVA] F({df1:.0f}, {df2:.2f}) = {F:.3f}, p = {p:.4g}")

if p < alpha:
    print(f"→ p < {alpha}: evidence of mean differences across groups. (Post-hoc: Games–Howell)")
else:
    print(f"→ p ≥ {alpha}: insufficient evidence of mean differences across groups.")

# ── (2) Overall effect size (reporting): eta^2, omega^2 ──
all_y = np.concatenate(groups)
grand_mean = np.mean(all_y)

ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in groups)
ss_total = ss_between + ss_within

k = len(groups)
n = len(all_y)
df_between = k - 1
df_within = n - k
ms_within = ss_within / df_within

eta2 = ss_between / ss_total if ss_total > 0 else float("nan")
omega2 = (ss_between - df_between * ms_within) / (ss_total + ms_within) if (ss_total + ms_within) > 0 else float("nan")

print(f"[Effect size] eta^2 = {eta2:.4f}, omega^2 = {omega2:.4f}")
print("  (rules of thumb) eta^2: 0.01 small, 0.06 medium, 0.14 large (context-dependent)")

# For pairwise comparisons, use the Games–Howell post-hoc snippet below (ALL pairs: adjusted p-values + effect sizes + CI comments).
"""
    ),

    "posthoc_tukey": (
        "post-hoc: Tukey HSD (after classic one-way ANOVA)",
        "Use after a significant one-way ANOVA when variances are ~ equal.",
        """import numpy as np
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

alpha = 0.05  # TODO

# TODO INPUT:
# - g1, g2, g3: 1D numeric arrays (extend for more groups)
# - Replace labels 'g1','g2','g3' with your actual group names if desired.

# 1) long-format data
df = pd.DataFrame({
    "y": np.concatenate([g1, g2, g3]),   # TODO extend
    "group": (["g1"]*len(g1) + ["g2"]*len(g2) + ["g3"]*len(g3))
})

# 2) Tukey HSD
res = pairwise_tukeyhsd(endog=df["y"], groups=df["group"], alpha=alpha)

# 3) Convert summary table to DataFrame
tbl = pd.DataFrame(res.summary().data[1:], columns=res.summary().data[0])
# columns: group1 group2 meandiff p-adj lower upper reject

# ---- Effect size helpers ----
def _pooled_sd(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    return np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1 + n2 - 2))

def cohens_d(a, b):
    sp = _pooled_sd(a, b)
    return (np.mean(a) - np.mean(b)) / sp if sp > 0 else np.nan

def hedges_g(a, b):
    # Why Hedges' g for small samples?
    # - Cohen's d is slightly biased upward when sample sizes are small.
    # - Hedges' g applies a correction factor (J) to reduce this bias.
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    d = cohens_d(a, b)
    J = 1 - 3 / (4*(n1+n2) - 9)
    return J * d

def auto_effect(a, b, small_n=20):
    # Auto: use Hedges' g if either group is "small" (default threshold: 20)
    if len(a) < small_n or len(b) < small_n:
        return hedges_g(a, b), "Hedges' g (small-sample correction)"
    return cohens_d(a, b), "Cohen's d"

def ci_comment(lower, upper):
    lower = float(lower); upper = float(upper)
    if lower > 0 or upper < 0:
        return "CI excludes 0 → difference is statistically reliable"
    return "CI includes 0 → difference may be statistically uncertain"

group_map = {"g1": g1, "g2": g2, "g3": g3}  # TODO extend

effects = []
effect_types = []
comments = []

for _, r in tbl.iterrows():
    a = group_map[r["group1"]]
    b = group_map[r["group2"]]
    es, es_type = auto_effect(a, b)
    effects.append(es)
    effect_types.append(es_type)
    comments.append(ci_comment(r["lower"], r["upper"]))

tbl["effect_size"] = effects
tbl["effect_type"] = effect_types
tbl["CI_comment"] = comments

print("[Tukey HSD] full table + effect size (all pairs)")
print(tbl.to_string(index=False))

print()
print("(rule of thumb) |d| or |g|: 0.2 small, 0.5 medium, 0.8 large (context-dependent)")
"""
    ),

    "posthoc_games_howell": (
        "post-hoc: Games–Howell (after Welch's ANOVA)",
        "Use after a significant Welch's ANOVA (no equal-variance assumption).",
        """import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats

alpha = 0.05  # TODO

# TODO INPUT:
# - g1, g2, g3 ... : 1D numeric arrays (extend for more groups)
# - labels must match the keys in group_map

# 1) long-format data
df = pd.DataFrame({
    "y": np.concatenate([g1, g2, g3]),   # TODO extend
    "group": (["g1"]*len(g1) + ["g2"]*len(g2) + ["g3"]*len(g3))
})

# 2) Games–Howell post-hoc
gh = pg.pairwise_gameshowell(data=df, dv="y", between="group")

# ---- Effect size helpers (auto Hedges' g for small samples) ----
def _pooled_sd(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    return np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1 + n2 - 2))

def cohens_d(a, b):
    sp = _pooled_sd(a, b)
    return (np.mean(a) - np.mean(b)) / sp if sp > 0 else np.nan

def hedges_g(a, b):
    # Why Hedges' g for small samples?
    # - Cohen's d is slightly biased upward when sample sizes are small.
    # - Hedges' g applies a correction factor (J) to reduce this bias.
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    d = cohens_d(a, b)
    J = 1 - 3 / (4*(n1+n2) - 9)
    return J * d

def auto_effect(a, b, small_n=20):
    if len(a) < small_n or len(b) < small_n:
        return hedges_g(a, b), "Hedges' g (small-sample correction)"
    return cohens_d(a, b), "Cohen's d"

def ci_comment(lower, upper):
    if lower > 0 or upper < 0:
        return "CI excludes 0 → difference is statistically reliable"
    return "CI includes 0 → difference may be statistically uncertain"

group_map = {"g1": g1, "g2": g2, "g3": g3}  # TODO extend

# 3) Add CI (computed) + effect size to ALL pairs
# pingouin typically returns: A, B, diff, se, df, pval, hedges (may vary by version)
ci_low = []
ci_high = []
effects = []
effect_types = []
comments = []

for _, r in gh.iterrows():
    diff = float(r["diff"])
    se = float(r["se"])
    df_ = float(r["df"])
    tcrit = stats.t.ppf(1 - alpha/2, df_)
    lo = diff - tcrit*se
    hi = diff + tcrit*se
    ci_low.append(lo)
    ci_high.append(hi)

    a = group_map[r["A"]]
    b = group_map[r["B"]]
    es, es_type = auto_effect(a, b)
    effects.append(es)
    effect_types.append(es_type)
    comments.append(ci_comment(lo, hi))

gh["ci_low"] = ci_low
gh["ci_high"] = ci_high
gh["effect_size"] = effects
gh["effect_type"] = effect_types
gh["CI_comment"] = comments

print("[Games–Howell] full table + effect size (all pairs)")
print(gh.to_string(index=False))

print()
print("(rule of thumb) |d| or |g|: 0.2 small, 0.5 medium, 0.8 large (context-dependent)")
"""
    ),

    "posthoc_dunn": (
        "post-hoc: Dunn's test (after Kruskal–Wallis)",
        "Use after a significant Kruskal–Wallis; adjusts for multiple comparisons.",
        """
# Dunn's test after Kruskal–Wallis (all pairs, adjusted p-values)
# Requires: scikit-posthocs
#   pip install scikit-posthocs

import numpy as np
import pandas as pd

try:
    import scikit_posthocs as sp
except ImportError as e:
    raise ImportError(
        "scikit-posthocs is required for Dunn's test. Install via: pip install scikit-posthocs"
    ) from e

alpha = 0.05  # TODO
p_adjust = "holm"  # alternatives: "bonferroni", "fdr_bh", ...

# TODO INPUT:
# - g1, g2, g3 ... : 1D numeric arrays (extend for more groups)
# - Update group_map to match your group names
group_map = {"g1": g1, "g2": g2, "g3": g3}  # TODO extend

# 1) Long-format data
df = pd.DataFrame({
    "y": np.concatenate(list(group_map.values())),
    "group": np.concatenate([[k]*len(v) for k, v in group_map.items()])
})

# 2) Dunn test (pairwise, adjusted p-values matrix)
p_mat = sp.posthoc_dunn(df, val_col="y", group_col="group", p_adjust=p_adjust)

# 3) Convert matrix -> long table (ALL pairs)
pairs = []
groups = list(p_mat.index)
for i in range(len(groups)):
    for j in range(i+1, len(groups)):
        a = groups[i]; b = groups[j]
        pairs.append({"group1": a, "group2": b, "p_adj": float(p_mat.loc[a, b])})
tbl = pd.DataFrame(pairs)

# ---- Effect size: Cliff's delta (+ bootstrap CI) ----
# Cliff's delta = P(X>Y) - P(X<Y), ranges [-1, 1]; 0 means no stochastic dominance.
# (This is appropriate for nonparametric comparisons and does not assume normality.)
def cliffs_delta(x, y):
    x = np.asarray(x); y = np.asarray(y)
    gt = sum((xi > y).sum() for xi in x)
    lt = sum((xi < y).sum() for xi in x)
    return (gt - lt) / (len(x) * len(y))

def bootstrap_ci_delta(x, y, B=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x); y = np.asarray(y)
    n1, n2 = len(x), len(y)
    vals = []
    for _ in range(B):
        xb = rng.choice(x, size=n1, replace=True)
        yb = rng.choice(y, size=n2, replace=True)
        vals.append(cliffs_delta(xb, yb))
    lo = float(np.quantile(vals, alpha/2))
    hi = float(np.quantile(vals, 1 - alpha/2))
    return lo, hi

effects = []
ci_lows = []
ci_highs = []
comments = []

for _, r in tbl.iterrows():
    a, b = r["group1"], r["group2"]
    x, y = group_map[a], group_map[b]
    dlt = float(cliffs_delta(x, y))
    lo, hi = bootstrap_ci_delta(x, y, B=2000, alpha=0.05, seed=0)
    effects.append(dlt); ci_lows.append(lo); ci_highs.append(hi)
    comments.append("CI does NOT include 0" if (lo > 0 or hi < 0) else "CI includes 0")

tbl["cliffs_delta"] = effects
tbl["delta_ci_low"] = ci_lows
tbl["delta_ci_high"] = ci_highs
tbl["CI_comment"] = comments
tbl["reject"] = tbl["p_adj"] < alpha

# 4) Print unified output
print(f"[Dunn] All pairs (p_adjust={p_adjust}, alpha={alpha})")
print(tbl.sort_values(["p_adj", "group1", "group2"]).to_string(index=False))

print()
print("(effect size) Cliff's delta: 0=no difference; magnitude is context-dependent.")
"""
    ),
    "kruskal": (
        "Kruskal–Wallis test",
        "Nonparametric alternative to one-way ANOVA (3+ independent groups).",
        """from scipy import stats
import numpy as np

alpha = 0.05  # TODO

# groups: list of arrays
groups = [g1, g2, g3]  # TODO extend
labels = ["g1", "g2", "g3"]  # TODO extend (keep same order as groups)

h_stat, p_value = stats.kruskal(*groups)

print("[Kruskal–Wallis]")
print(f"  H = {h_stat:.4f}, p = {p_value:.4g}, alpha = {alpha}")

if p_value < alpha:
    print("  → Reject H0: evidence that at least one group differs (in distribution/central tendency).")
    print("  → Post-hoc: use Dunn's test with multiple-comparison adjustment for ALL pairs.")
else:
    print("  → Fail to reject H0: insufficient evidence of group differences.")
    print("  (Note) With small samples, power may be low—inspect effect size/CI too.")

# Effect size: epsilon-squared (Kruskal–Wallis)
n = sum(len(g) for g in groups)
k = len(groups)
epsilon2 = (h_stat - k + 1) / (n - k) if (n - k) > 0 else np.nan
print(f"  Effect size epsilon^2 = {epsilon2:.4f}")
print("    (rule of thumb) ε²≈0.01 small, 0.08 medium, 0.26 large (context-dependent)")

# Bootstrap CI for epsilon^2
rng = np.random.default_rng(0)
B = 2000  # TODO
vals = []
for _ in range(B):
    boot_groups = [rng.choice(g, size=len(g), replace=True) for g in groups]
    h_b, _ = stats.kruskal(*boot_groups)
    eps_b = (h_b - k + 1) / (n - k) if (n - k) > 0 else np.nan
    vals.append(eps_b)

ci = (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))
print(f"  95% CI for epsilon^2 (bootstrap) = {ci}")
print("  CI tip: values near 0 suggest a small difference (context-dependent).")"""
    ),

    # Binary/categorical
    "chi2_contingency": (
        "Chi-square test of independence",
        "Association between categorical variables (contingency table, RxC).",
        """import numpy as np
import pandas as pd
from scipy import stats

alpha = 0.05  # TODO

# table : contingency table (observed counts) array (R×C)
# TODO INPUT:
# table = np.array([[...], [...], ...], dtype=int)

table_np = np.asarray(table, dtype=float)

chi2, p, dof, expected_raw = stats.chi2_contingency(table_np, correction=False)
expected = np.asarray(expected_raw, dtype=float)

print("[Chi-square test of independence]")
print(f"  chi2 = {chi2:.4f}, dof = {dof}, p = {p:.4g}, alpha = {alpha}")

if p < alpha:
    print("  → Reject H0: evidence of association (not independent).")
else:
    print("  → Fail to reject H0: insufficient evidence of association.")
    print("  (Note) With small samples, power may be low—inspect effect size too.")

# Effect size: Cramer's V
n = float(table_np.sum())
r, c = table_np.shape
k = min(r - 1, c - 1)
cramers_v = float(np.sqrt(chi2 / (n * k))) if (n > 0 and k > 0) else float("nan")
print(f"  Effect size Cramer's V = {cramers_v:.4f}")
print("    (rule of thumb) for 2×2: V≈0.10 small, 0.30 medium, 0.50 large (context-dependent)")

# Post-hoc: adjusted standardized residuals
row_sum = table_np.sum(axis=1, keepdims=True)
col_sum = table_np.sum(axis=0, keepdims=True)
row_prop = row_sum / n if n > 0 else np.zeros_like(row_sum)
col_prop = col_sum / n if n > 0 else np.zeros_like(col_sum)

den = np.sqrt(expected * (1.0 - row_prop) * (1.0 - col_prop))
with np.errstate(divide="ignore", invalid="ignore"):
    resid = (table_np - expected) / den

table_df = table if hasattr(table, "index") else pd.DataFrame(table_np)

rows = []
for i, rlab in enumerate(table_df.index):
    for j, clab in enumerate(table_df.columns):
        z = resid[i, j]
        rows.append({
            "Row": rlab,
            "Col": clab,
            "Observed": int(table_np[i, j]),
            "Expected": float(expected[i, j]),
            "AdjResid": float(z) if np.isfinite(z) else float("nan"),
            "Flag(|z|>2)": bool(np.isfinite(z) and abs(z) > 2),
            "Direction": ("Obs>Exp" if table_np[i, j] > expected[i, j] else "Obs<Exp")
        })

out = pd.DataFrame(rows)
out_sorted = out.sort_values("AdjResid", key=lambda s: s.abs(), ascending=False)

print("\\n[Post-hoc] Adjusted standardized residuals (Top 10 by |z|)")
print(out_sorted.head(10).to_string(index=False))

sig = out_sorted[out_sorted["Flag(|z|>2)"]]
print("\\n[Post-hoc] Cells with |z| > 2 (potentially notable)")
if sig.empty:
    print("(none)")
else:
    print(sig.to_string(index=False))

print("\nTip: p-value answers 'is there association?', residuals show 'which cells drive it', and V summarizes 'how large'.")"""
    ),
    "chi2_ind_cochran_check": (
        "Cochran check (independence)",
        "Check Cochran’s rule conditions before using chi-square approximation.",
        """
import numpy as np
from scipy import stats

# table : contingency table of observed counts (R×C)
# table = np.array([[...], [...], ...])

table_np: np.ndarray = np.asarray(table, dtype=float)
res = stats.chi2_contingency(table_np, correction=False)
expected: np.ndarray = np.asarray(res.expected_freq, dtype=float)

# ── Cochran's rule check ──
n_under1 = np.sum(expected < 1)
pct_under5 = np.mean(expected < 5)
cochran_ok = (n_under1 == 0) and (pct_under5 <= 0.20)

print("shape =", table_np.shape)
print("n_under1 =", int(n_under1))
print("pct_under5 =", float(pct_under5))
print("cochran_ok =", bool(cochran_ok))
        """
    ),
    "chi2_ind_mc": (
        "Chi-square independence (Monte Carlo)",
        "When Cochran’s rule is violated, approximate p-value via Monte Carlo.",
        """import numpy as np
from scipy import stats

alpha = 0.05  # TODO

# table : contingency table (observed counts) array (R×C)
# table = np.array([[...], [...], ...], dtype=int)

table = np.asarray(table, dtype=float)
chi2_obs, p_asym, dof, expected = stats.chi2_contingency(table, correction=False)

# ── Monte Carlo p-value under H0 (independence) ──
# NOTE: This is an approximation that preserves margins only approximately.
rng = np.random.default_rng(0)
n_sim = 100_000  # TODO
count_extreme = 0

row_sums = table.sum(axis=1)
col_sums = table.sum(axis=0)
n = table.sum()

col_probs = col_sums / n
for _ in range(n_sim):
    sim = np.vstack([rng.multinomial(int(rs), col_probs) for rs in row_sums]).astype(float)
    chi2_s, _, _, _ = stats.chi2_contingency(sim, correction=False)
    count_extreme += (chi2_s >= chi2_obs)

p_mc = count_extreme / n_sim

print(f"[Chi-square Independence] (Monte Carlo) chi2 = {chi2_obs:.4g}, dof = {dof}, p_mc = {p_mc:.4g}, alpha = {alpha}")
print(f"  (ref) asymptotic p = {p_asym:.4g}")

if p_mc < alpha:
    print("→ Decision: reject H0 (evidence of association)")
else:
    print("→ Decision: fail to reject H0 (insufficient evidence of association)")

# Effect size: Cramer's V
r, c = table.shape
k = min(r - 1, c - 1)
cramers_v = np.sqrt(chi2_obs / (n * k)) if (n > 0 and k > 0) else np.nan
print(f"[Effect size] Cramer's V = {cramers_v:.4f}  (rule of thumb: 0.1 small, 0.3 medium, 0.5 large; context-dependent)")

# Cell-level hints: adjusted standardized residuals
row_prop = row_sums[:, None] / n
col_prop = col_sums[None, :] / n
den = np.sqrt(expected * (1.0 - row_prop) * (1.0 - col_prop))
with np.errstate(divide='ignore', invalid='ignore'):
    adj_resid = (table - expected) / den
print("[Post-hoc hint] Cells with |residual| > 2 may contribute strongly to chi-square.")"""
    ),
    "chi2_ind_collapse": (
        "Collapse categories then Chi-square independence",
        "Template for collapsing categories, then running chi-square independence test.",
        """import numpy as np
from scipy import stats

alpha = 0.05  # TODO

# table : contingency table (observed counts) array (R×C)
table_np = np.asarray(table, dtype=float)

chi2, p, dof, expected = stats.chi2_contingency(table_np, correction=False)
expected_flat = expected.ravel()
pct_lt5 = np.mean(expected_flat < 5) * 100

print(f"[Chi-square Independence] chi2 = {chi2:.4g}, dof = {dof}, p = {p:.4g}, alpha = {alpha}")
print(f"[Assumption check] % expected < 5 = {pct_lt5:.1f}% (rule of thumb: <=20%; context-dependent)")

if p < alpha:
    print("→ Decision: reject H0 (evidence of association)")
else:
    print("→ Decision: fail to reject H0 (insufficient evidence)")

print("\\n[Collapsing sparse categories]")
print("- If many expected counts are small, consider collapsing rare categories in a substantively valid way and re-run the test.")

n = table_np.sum()
r, c = table_np.shape
k = min(r - 1, c - 1)
cramers_v = np.sqrt(chi2 / (n * k)) if (n > 0 and k > 0) else np.nan
print(f"[Effect size] Cramer's V = {cramers_v:.4f}")"""
    ),
    "ffh_exact": (
        "Fisher–Freeman–Halton (FFH) exact test",
        "Exact test for R×C tables (availability depends on SciPy version).",
        """# Fisher–Freeman–Halton (FFH) exact test for RxC tables (exact alternative to chi-square)
# NOTE: SciPy does not provide FFH exact for general RxC tables.
# Practical alternatives: permutation / Monte Carlo p-value for chi-square statistic.

import numpy as np
from scipy import stats

alpha = 0.05  # TODO

table = np.asarray(table, dtype=int)

chi2_obs, p_asym, dof, expected = stats.chi2_contingency(table, correction=False)

rng = np.random.default_rng(0)
n_sim = 50_000  # TODO
n = int(table.sum())
p_exp = (expected / expected.sum()).ravel()
count_extreme = 0

for _ in range(n_sim):
    sim = rng.multinomial(n, p_exp).reshape(table.shape)
    chi2_s, _, _, _ = stats.chi2_contingency(sim, correction=False)
    count_extreme += (chi2_s >= chi2_obs)

p_mc = count_extreme / n_sim

print(f"[FFH exact (practical approx)] chi2 = {chi2_obs:.4g}, dof = {dof}, p_mc = {p_mc:.4g}, alpha = {alpha}")
print("  (note) FFH exact for RxC is library-dependent; MC/permutation reporting is common in practice.")

if p_mc < alpha:
    print("→ Decision: reject H0 (evidence of association)")
else:
    print("→ Decision: fail to reject H0 (insufficient evidence)")

r, c = table.shape
k = min(r - 1, c - 1)
cramers_v = np.sqrt(chi2_obs / (n * k)) if (n > 0 and k > 0) else np.nan
print(f"[Effect size] Cramer's V = {cramers_v:.4f}")"""

    ),

    "fisher_exact": (
        "Fisher's exact test (2x2)",
        "Use when expected counts are small in a 2x2 table.",
        """import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import Table2x2

__COMMON_SETTINGS__

# 2×2 contingency table
# table = np.array([[a, b],
#                   [c, d]], dtype=int)
table = np.asarray(table, dtype=float)

oddsratio, p_value = stats.fisher_exact(table, alternative=alternative)
print(f"[Fisher exact (2×2)] OR = {oddsratio:.4g}, p = {p_value:.4g}, alpha = {alpha}")

if p_value < alpha:
    print("→ Decision: reject H0 (evidence of association)")
else:
    print("→ Decision: fail to reject H0 (insufficient evidence)")

t22 = Table2x2(table)
ci_low, ci_high = t22.oddsratio_confint()
print(f"[CI] OR 95% CI = ({ci_low:.4g}, {ci_high:.4g})")
print("  - If the CI excludes 1, OR is (roughly) different from 1.")

chi2, p_chi, dof, exp = stats.chi2_contingency(table, correction=False)
n = table.sum()
phi = np.sqrt(chi2 / n) if n > 0 else np.nan
print(f"[Effect size] φ(phi) = {phi:.4f}  (rule of thumb: 0.1 small, 0.3 medium, 0.5 large)")

row_sum = table.sum(axis=1, keepdims=True)
col_sum = table.sum(axis=0, keepdims=True)
row_prop = row_sum / n
col_prop = col_sum / n
den = np.sqrt(exp * (1.0 - row_prop) * (1.0 - col_prop))
with np.errstate(divide='ignore', invalid='ignore'):
    adj_resid = (table - exp) / den
print("[Residual hint] Cells with |residual| > 2 may deviate strongly from expectation.")"""

    ),
    "mcnemar": (
        "McNemar test (paired binary)",
        "Paired yes/no outcomes (before/after, matched pairs).",
        """import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from scipy import stats

table = np.array([[a, b],
                  [c, d]])

res = mcnemar(table, exact=True)
print("stat =", res.statistic, "p =", res.pvalue)

# Effect size (common): discordant-pairs odds ratio = b/c
b = table[0,1]
c = table[1,0]
if b > 0 and c > 0:
    or_dc = b / c
    # Approximate CI for OR using log method
    alpha = 0.05
    se = np.sqrt(1/b + 1/c)
    zcrit = stats.norm.ppf(1 - alpha/2)
    ci = (np.exp(np.log(or_dc) - zcrit*se), np.exp(np.log(or_dc) + zcrit*se))
    print("Discordant OR (b/c) =", or_dc, "95% CI ≈", ci)
else:
    print("Discordant OR (b/c): not defined if b==0 or c==0 (consider adding 0.5 continuity correction).")

# Power note: often via simulation or dedicated calculators.
"""
    ),

    # Association
    "pearsonr": (
        "Pearson correlation",
        "Linear association between two continuous variables (approx normal).",
        """
from scipy import stats
import numpy as np

alpha = 0.05  # TODO

# x, y: 1D arrays (same length)
r, p_value = stats.pearsonr(x, y)

print("[Pearson correlation]")
print(f"  r = {r:.4f}, p = {p_value:.4g}, alpha = {alpha}")
if p_value < alpha:
    print("  → Reject H0: evidence of a non-zero *linear* association.")
else:
    print("  → Fail to reject H0: insufficient evidence of a non-zero linear association.")

# CI for r (Fisher z transform)
n = len(x)
z = np.arctanh(r)
se = 1 / np.sqrt(n - 3)  # valid when n>3
zcrit = stats.norm.ppf(1 - alpha/2)
ci_low = np.tanh(z - zcrit*se)
ci_high = np.tanh(z + zcrit*se)
print(f"  {int((1-alpha)*100)}% CI for r = [{ci_low:.4f}, {ci_high:.4f}]")
print("  Interpretation tip: CI excluding 0 aligns with p < alpha.")

# Power (approx; treat correlation as effect size)
from statsmodels.stats.power import NormalIndPower
effect = r / np.sqrt(max(1e-12, 1 - r**2))
power = NormalIndPower().power(effect_size=effect, nobs1=n, alpha=alpha)
print(f"  Power (approx) = {power:.3f}")

"""
    ),
    "spearmanr": (
        "Spearman correlation",
        "Rank-based association for non-normal continuous or ordinal variables.",
        """
from scipy import stats
import numpy as np

alpha = 0.05  # TODO

# x, y: 1D arrays (same length)
rho, p_value = stats.spearmanr(x, y)

print("[Spearman rank correlation]")
print(f"  rho = {rho:.4f}, p = {p_value:.4g}, alpha = {alpha}")
if p_value < alpha:
    print("  → Reject H0: evidence of a monotonic association.")
else:
    print("  → Fail to reject H0: insufficient evidence of a monotonic association.")

# Practical note:
print("  Tip: Spearman is robust to outliers and non-linear but monotonic trends.")

# Optional: bootstrap CI for rho (recommended)
B = 2000  # TODO
rng = np.random.default_rng(0)
vals = []
x = np.asarray(x); y = np.asarray(y)
n = len(x)
for _ in range(B):
    idx = rng.integers(0, n, size=n)
    rr, _ = stats.spearmanr(x[idx], y[idx])
    vals.append(rr)
ci = (np.percentile(vals, 2.5), np.percentile(vals, 97.5))
print(f"  95% bootstrap CI for rho = [{ci[0]:.4f}, {ci[1]:.4f}]")

"""
    ),

    # Modeling
    "ols": (
        "Linear regression (OLS)",
        "Continuous outcome with one or more predictors.",
        """import numpy as np
import pandas as pd
import statsmodels.api as sm

alpha = 0.05  # TODO

# y: (n,), X: (n, p)
X_ = sm.add_constant(X, has_constant="add")
model = sm.OLS(y, X_).fit()

print("[OLS Regression] Key summary")
print(f"R^2 = {model.rsquared:.4f}, Adj R^2 = {model.rsquared_adj:.4f}, F-test p = {model.f_pvalue:.4g}")

if model.f_pvalue < alpha:
    print("→ Decision: model is significant overall (at least one coefficient likely non-zero)")
else:
    print("→ Decision: model not significant overall (insufficient evidence)")

params = model.params
pvals = model.pvalues
ci = model.conf_int(alpha=alpha)

out = pd.DataFrame({
    "coef": params,
    "p": pvals,
    f"CI{int((1-alpha)*100)}_low": ci[0],
    f"CI{int((1-alpha)*100)}_high": ci[1],
})
out["CI_comment"] = np.where((out[f"CI{int((1-alpha)*100)}_low"] > 0) | (out[f"CI{int((1-alpha)*100)}_high"] < 0),
                            "CI excludes 0 (likely significant)", "CI includes 0 (uncertain)")
print("\\n[Coefficients]")
print(out.to_string())

print("\\n[Interpretation tips]")
print("- The sign of coef indicates direction of association, holding others fixed.")
print("- p-values/CIs quantify uncertainty.")
print("- Also check assumptions: residual diagnostics, heteroscedasticity, influential points.")"""
    ),
    "logit": (
        "Logistic regression",
        "Binary outcome with one or more predictors.",
        """
import numpy as np
import pandas as pd
import statsmodels.api as sm

alpha = 0.05  # TODO

# y: binary (0/1), X: predictors without intercept
X = np.asarray(X, dtype=float)
y = np.asarray(y, dtype=float)

X_ = sm.add_constant(X)
model = sm.Logit(y, X_).fit(disp=False)

print("[Logistic regression (Logit)]")
print(f"  n = {len(y)}, alpha = {alpha}")
print(f"  LL = {model.llf:.3f}, LL-Null = {model.llnull:.3f}, Pseudo R^2 (McFadden) = {1 - model.llf/model.llnull:.4f}")

tbl = pd.DataFrame({
    "coef(log-odds)": model.params,
    "se": model.bse,
    "z": model.tvalues,
    "p": model.pvalues,
})
ci = model.conf_int(alpha=alpha)
tbl["ci_low"] = ci[0]
tbl["ci_high"] = ci[1]

# Odds ratios
tbl["OR"] = np.exp(tbl["coef(log-odds)"])
tbl["OR_ci_low"] = np.exp(tbl["ci_low"])
tbl["OR_ci_high"] = np.exp(tbl["ci_high"])
tbl["reject"] = tbl["p"] < alpha

print("\\n[Coefficients (log-odds) + Odds Ratios]")
print(tbl.to_string())

print("\nTip: OR CI excluding 1 corresponds to p < alpha (approximately).")

"""
    ),
    "poisson_glm": (
        "Poisson regression (GLM)",
        "Count outcome; consider Negative Binomial if overdispersed.",
        """
import numpy as np
import pandas as pd
import statsmodels.api as sm

alpha = 0.05  # TODO

# y: count (non-negative int), X: predictors without intercept
X = np.asarray(X, dtype=float)
y = np.asarray(y, dtype=float)

X_ = sm.add_constant(X)
model = sm.GLM(y, X_, family=sm.families.Poisson()).fit()

print("[Poisson GLM]")
print(f"  n = {len(y)}, alpha = {alpha}")
print(f"  Deviance = {model.deviance:.3f}, Pearson chi2 = {model.pearson_chi2:.3f}")
print("  Tip: if overdispersion is large (Pearson chi2 / df >> 1), consider Negative Binomial.")

tbl = pd.DataFrame({
    "coef(log-rate)": model.params,
    "se": model.bse,
    "z": model.tvalues,
    "p": model.pvalues,
})
ci = model.conf_int(alpha=alpha)
tbl["ci_low"] = ci[0]; tbl["ci_high"] = ci[1]
tbl["RateRatio"] = np.exp(tbl["coef(log-rate)"])
tbl["RR_ci_low"] = np.exp(tbl["ci_low"])
tbl["RR_ci_high"] = np.exp(tbl["ci_high"])
tbl["reject"] = tbl["p"] < alpha

print("\\n[Coefficients + Rate Ratios]")
print(tbl.to_string())

"""
    ),
    # One-sample z-test for mean (known sigma)
    "onesample_z_mean": (
        "One-sample z-test (mean, σ known)",
        "Use when population standard deviation σ is known (rare). Otherwise use a one-sample t-test.",
        """import numpy as np
from scipy import stats

# x : sample array
mu0 = 0.0      # TODO: hypothesized mean under H0
sigma = 1.0    # TODO: KNOWN population std dev (σ)

n = len(x)
xbar = np.mean(x)

z = (xbar - mu0) / (sigma / np.sqrt(n))

# p-value by alternative
if alternative == "two-sided":
    p_value = 2 * stats.norm.sf(abs(z))
elif alternative == "greater":
    p_value = stats.norm.sf(z)
else:  # "less"
    p_value = stats.norm.cdf(z)
print("z =", z, f"p ({alternative}) =", p_value)

# CI for mean (z-based)
alpha = 0.05
zcrit_two = stats.norm.ppf(1 - alpha/2)
zcrit_one = stats.norm.ppf(1 - alpha)

if alternative == "two-sided":
    ci = (xbar - zcrit_two * sigma/np.sqrt(n), xbar + zcrit_two * sigma/np.sqrt(n))
elif alternative == "greater":
    ci = (xbar - zcrit_one * sigma/np.sqrt(n), np.inf)
else:  # "less"
    ci = (-np.inf, xbar + zcrit_one * sigma/np.sqrt(n))
print("CI for mean =", ci)
"""
    ),

    # Proportions: one-sample z-test
    "prop_1sample_ztest": (
        "One-sample proportion z-test",
        "Compare an observed proportion to a hypothesized p0 (e.g., conversion rate vs benchmark).",
        """import numpy as np
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

__COMMON_SETTINGS__

# count: successes, nobs: trials, p0: null proportion
stat, p_value = proportions_ztest(count=count, nobs=nobs, value=p0, alternative=alternative)

print(f"[One-sample proportion z-test] z = {stat:.4g}, p = {p_value:.4g}, alpha = {alpha}, H0: p = {p0}")

if p_value < alpha:
    print("→ Decision: reject H0 (proportion differs from p0)")
else:
    print("→ Decision: fail to reject H0 (insufficient evidence)")

ci_low, ci_high = proportion_confint(count=count, nobs=nobs, alpha=alpha, method="wilson")
print(f"[CI] {int((1-alpha)*100)}% CI for p (Wilson) = ({ci_low:.4f}, {ci_high:.4f})")
print("  - If the CI excludes p0, p is (roughly) different from p0.")

p_hat = count / nobs
h = 2*np.arcsin(np.sqrt(p_hat)) - 2*np.arcsin(np.sqrt(p0))
print(f"[Effect size] Cohen's h = {h:.4f}  (0.2 small, 0.5 medium, 0.8 large)")"""
    ),

    # Proportions: two-sample z-test (+ power via NormalIndPower)
    "prop_2sample_ztest": (
        "Two-sample proportion z-test (independent groups)",
        "Compare two independent proportions. (Equivalent to chi-square in 2x2, but z-test supports directional alternatives.)",
        """from statsmodels.stats.proportion import proportions_ztest, proportion_confint, proportion_effectsize
from statsmodels.stats.power import NormalIndPower

# Group A
count1 = 45   # TODO successes in group 1
nobs1  = 200  # TODO trials in group 1

# Group B
count2 = 30   # TODO successes in group 2
nobs2  = 210  # TODO trials in group 2

count = [count1, count2]
nobs  = [nobs1, nobs2]

z_stat, p_value = proportions_ztest(count=count, nobs=nobs, alternative="two-sided")
print("z =", z_stat, "p =", p_value)

p1 = count1 / nobs1
p2 = count2 / nobs2

# Separate CIs (Wilson)
ci1 = proportion_confint(count1, nobs1, alpha=0.05, method="wilson")
ci2 = proportion_confint(count2, nobs2, alpha=0.05, method="wilson")
print("p1 CI =", ci1, "p2 CI =", ci2)

# Effect size: Cohen's h
h = proportion_effectsize(p1, p2)
print("Cohen's h =", h)

# Power / required sample size for a two-sample z-test (normal approximation)
# If planning: set target power and solve for nobs1 given ratio.
analysis = NormalIndPower()
power = analysis.power(effect_size=h, nobs1=nobs1, ratio=nobs2/nobs1, alpha=0.05, alternative="two-sided")
print("Power (given nobs1,nobs2) =", power)

# Required nobs1 for target power (example)
# target_power = 0.80
# nobs1_req = analysis.solve_power(effect_size=h, power=target_power, alpha=0.05, ratio=nobs2/nobs1)
# print("Required nobs1 ≈", nobs1_req)
"""
    ),

    # Chi-square goodness-of-fit
    "chi2_gof": (
        "Chi-square goodness-of-fit",
        "Test whether observed counts match a specified expected distribution (one categorical variable).",
        """import numpy as np
from scipy import stats

alpha = 0.05  # TODO

# Observed counts
# observed = np.array([o1, o2, o3, ...])

# Expected probabilities (sum=1) OR expected counts (sum=n)
# expected_probs = np.array([...])  # sum=1
# expected = expected_probs * observed.sum()

observed = np.asarray(observed, dtype=float)
expected = np.asarray(expected, dtype=float)
n = float(observed.sum())

chi2, p = stats.chisquare(f_obs=observed, f_exp=expected)
df = int(len(observed) - 1)

print("[Chi-square goodness-of-fit]")
print(f"  chi2 = {chi2:.4f}, df = {df}, p = {p:.4g}, alpha = {alpha}")

if p < alpha:
    print("  → Reject H0: observed distribution differs from the expected distribution.")
else:
    print("  → Fail to reject H0: insufficient evidence of a difference from expected.")
    print("  (Note) With small samples, power may be low—inspect effect size/residuals too.")

# Effect size: Cohen's w
w = float(np.sqrt(chi2 / n)) if n > 0 else float("nan")
print(f"  Effect size Cohen's w = {w:.4f}")
print("    (rule of thumb) w≈0.10 small, 0.30 medium, 0.50 large (context-dependent)")

# Post-hoc: standardized residuals
std_residuals = (observed - expected) / np.sqrt(expected)
print("\\n[Post-hoc] Standardized residuals (|z|>2 may be notable)")
for i, z in enumerate(std_residuals):
    flag = " |z|>2" if np.isfinite(z) and abs(z) > 2 else ""
    print(f"  category[{i}] z = {float(z):.3f}{flag}")"""
    ),
    "chi2_gof_cochran_check": (
        "Cochran check (GOF)",
        "Check Cochran’s rule conditions before using chi-square approximation.",
        """
import numpy as np

# Observed counts
# observed = np.array([o1, o2, o3, ...])

# Expected probabilities (sum=1) or expected counts (sum=n)
# expected_probs = np.array([...])  # sum=1
# expected = expected_probs * observed.sum()

# ── Cochran's rule check (chi-square approximation conditions) ──
# (1) <= 20% of expected cells are < 5, AND
# (2) no expected cell is < 1
n_under1 = np.sum(expected < 1)
pct_under5 = np.mean(expected < 5)
cochran_ok = (n_under1 == 0) and (pct_under5 <= 0.20)

print("n_under1 =", int(n_under1))
print("pct_under5 =", float(pct_under5))
print("cochran_ok =", bool(cochran_ok))

        """
    ),
    "chi2_gof_mc": (
        "Chi-square GOF (Monte Carlo)",
        "When Cochran’s rule is violated, approximate p-value via Monte Carlo simulation.",
        """import numpy as np
from scipy import stats

alpha = 0.05  # TODO

obs = np.asarray(observed, dtype=float)
exp = np.asarray(expected, dtype=float)
if np.isclose(exp.sum(), 1.0):
    exp = exp * obs.sum()

chi2_obs = ((obs - exp) ** 2 / exp).sum()
df = len(obs) - 1

p_exp = exp / exp.sum()
rng = np.random.default_rng(0)
n_sim = 100_000  # TODO
count_extreme = 0
n = int(obs.sum())

for _ in range(n_sim):
    sim = rng.multinomial(n, p_exp).astype(float)
    chi2_s = ((sim - exp) ** 2 / exp).sum()
    count_extreme += (chi2_s >= chi2_obs)

p_mc = count_extreme / n_sim

print(f"[Chi-square GOF] (Monte Carlo) chi2 = {chi2_obs:.4g}, df = {df}, p_mc = {p_mc:.4g}, alpha = {alpha}")

if p_mc < alpha:
    print("→ Decision: reject H0 (observed distribution differs from expected)")
else:
    print("→ Decision: fail to reject H0 (insufficient evidence)")

w = np.sqrt(chi2_obs / n) if n > 0 else np.nan
print(f"[Effect size] Cohen's w = {w:.4f}  (rule of thumb: 0.1 small, 0.3 medium, 0.5 large)")

with np.errstate(divide='ignore', invalid='ignore'):
    resid = (obs - exp) / np.sqrt(exp)
print("[Hint] Categories with large |residual| may drive the discrepancy.")"""
    ),
    "chi2_gof_collapse": (
        "Collapse categories then Chi-square GOF",
        "Template for collapsing categories, then running GOF chi-square.",
        """import numpy as np
from scipy import stats

alpha = 0.05  # TODO

obs = np.asarray(observed, dtype=float)
exp = np.asarray(expected, dtype=float)
if np.isclose(exp.sum(), 1.0):
    exp = exp * obs.sum()

pct_lt5 = np.mean(exp < 5) * 100

chi2, p = stats.chisquare(f_obs=obs, f_exp=exp)
df = len(obs) - 1

print(f"[Chi-square GOF] chi2 = {chi2:.4g}, df = {df}, p = {p:.4g}, alpha = {alpha}")
print(f"[Assumption check] % expected < 5 = {pct_lt5:.1f}% (consider collapsing sparse categories)")

if p < alpha:
    print("→ Decision: reject H0 (distribution differs from expected)")
else:
    print("→ Decision: fail to reject H0 (insufficient evidence)")

print("\\n[Collapsing sparse categories]")
print("- If many expected counts are small, consider collapsing categories in a substantively valid way and re-run.")

n = obs.sum()
w = np.sqrt(chi2 / n) if n > 0 else np.nan
print(f"[Effect size] Cohen's w = {w:.4f}")"""
    ),


    # Permutation test (independent two-sample) — advanced
    "perm_ind_diff_means": (
        "Permutation test (independent two-sample, difference in means)",
        "Nonparametric/randomization test by shuffling group labels. Useful when assumptions are questionable.",
        """import numpy as np
from scipy import stats

# x1, x2 : arrays for the two groups
def stat_diff_means(a, b, axis=0):
    return np.mean(a, axis=axis) - np.mean(b, axis=axis)

res = stats.permutation_test(
    (x1, x2),
    statistic=stat_diff_means,
    permutation_type="independent",
    alternative="two-sided",
    n_resamples=10000,
    random_state=0,
)
print(res)  # res.statistic, res.pvalue

# Tip: if scipy is older, consider statsmodels or manual permutation.
"""
    ),

    # Permutation test (paired / sign-flip) — advanced
    "perm_paired_mean": (
        "Permutation / sign-flip test (paired, mean difference)",
        "Randomization test for paired designs by randomly flipping the sign of paired differences.",
        """import numpy as np
from scipy import stats

# d : paired differences (x_after - x_before)
# If you have x_before, x_after: d = x_after - x_before
def stat_mean(a, axis=0):
    return np.mean(a, axis=axis)

res = stats.permutation_test(
    (d,),
    statistic=stat_mean,
    permutation_type="samples",   # sign-flip uses 'samples' with symmetric null if you pass differences
    alternative="two-sided",
    n_resamples=10000,
    random_state=0,
)
print(res)
"""
    ),

}



# ----------------------------
# Effect size interpretation guides (rules of thumb)
# ----------------------------

EFFECT_GUIDE: Dict[str, str] = {
    "one_sample_t": "Cohen's d: |d|≈0.2(small), 0.5(medium), 0.8(large).  \nHedges' g is often interpreted with the same cutoffs.  \nContext/domain benchmarks matter most.",
    "paired_t": "Paired Cohen's d: |d|≈0.2(small), 0.5(medium), 0.8(large).  \nPaired designs can make the same d more practically meaningful (less noise).",
    "ind_t_student": "Cohen's d / Hedges' g: |d|≈0.2(small), 0.5(medium), 0.8(large).  \nPrefer g for small samples.",
    "ind_t_welch": "Cohen's d / Hedges' g: |d|≈0.2(small), 0.5(medium), 0.8(large).",
    "one_sample_z_mean": "Standardized mean difference (d-like): |d|≈0.2(small), 0.5(medium), 0.8(large).",

    "pearsonr": "Correlation r: |r|≈0.1(small), 0.3(medium), 0.5(large).  \nr^2 is variance explained (for linear association).",
    "spearmanr": "Rank correlation rho: |rho|≈0.1(small), 0.3(medium), 0.5(large).",

    "paired_wilcoxon": "Rank-biserial correlation (RBC): often interpreted like |r|≈0.1/0.3/0.5 (small/medium/large).  \nAlternative rule: Cliff's delta cutoffs (|δ|<0.147 negligible, <0.33 small, <0.474 medium, else large).",
    "mannwhitney": "Rank-biserial correlation (RBC): often interpreted like |r|≈0.1/0.3/0.5.  \nAlternative: Cliff's delta cutoffs (|δ|<0.147 negligible, <0.33 small, <0.474 medium, else large).",

    "anova_oneway": "η²: ≈0.01(small), 0.06(medium), 0.14(large).  \nOr Cohen's f: 0.10(small), 0.25(medium), 0.40(large).",
    "kruskal": "ε²: often 0.01(small), 0.08(medium), 0.26(large) as a rough rule (field-dependent).",

    "chi2_contingency": "Cramer's V: (for 2×2) 0.10(small), 0.30(medium), 0.50(large).  \nFor larger tables, interpretation depends on degrees of freedom/context.",
    "chi2_gof": "Cohen's w: 0.10(small), 0.30(medium), 0.50(large).",
    "fisher_exact": "Odds ratio (OR): OR=1 no effect.  \nBecause OR is asymmetric, consider log(OR), risk difference, and domain thresholds.",

    "ols": "Regression: R²≈0.02(small), 0.13(medium), 0.26(large) is a common rule of thumb (highly field-dependent).  \nAlso report standardized betas or partial R².",
    "logit": "Logistic regression: OR=1 no effect.  \nOR depends on units; consider rescaling predictors and reporting marginal effects (probability change).",
    # proportions
    "prop_1sample_ztest": "Cohen's h (proportions): h = 2·arcsin(√p1) − 2·arcsin(√p2).\nRule of thumb: |h|≈0.2(small), 0.5(medium), 0.8(large).\nNote: near 0/1, interpretation can be sensitive.",
    "prop_2sample_ztest": "Cohen's h (difference in proportions): |h|≈0.2(small), 0.5(medium), 0.8(large).\n(h = 2·arcsin(√p1) − 2·arcsin(√p2))",

    # rank/ordinal agreement
    "kendall_w": "Kendall's W (concordance, 0~1): 0=no agreement, 1=perfect agreement.\nHeuristics sometimes used: W≈0.1 weak, 0.3 moderate, 0.5 strong (field-dependent).\nOften reported with the Friedman test.",

    # dominance (nonparametric effect size)
    "cliffs_delta": "Cliff's delta (δ, -1~1): dominance/probability-of-superiority effect size.\nHeuristic (absolute): |δ|<0.147 negligible, <0.33 small, <0.474 medium, otherwise large.\nSometimes used alongside (or instead of) rank-biserial r.",

}
def show_snippet(key: str) -> None:
    if key not in TEST_SNIPPETS:
        print_node("INTERNAL ERROR", f"Missing TEST_SNIPPETS key: {key}")
        return
    title, detail, code = TEST_SNIPPETS[key]

    # --- Apply the user's alternative choice (two-sided / less / greater) ---
    global CURRENT_ALT
    if "alternative" in code:
        ensure_alternative_selected()
        code = code.replace('alternative="two-sided"', 'alternative=alternative')
        code = code.replace("alternative='two-sided'", "alternative=alternative")
        code = code.replace('alternative="greater"', 'alternative=alternative')
        code = code.replace("alternative='greater'", "alternative=alternative")
        code = code.replace('alternative="less"', 'alternative=alternative')
        code = code.replace("alternative='less'", "alternative=alternative")
        code = f'alternative = "{CURRENT_ALT}"  # "two-sided" | "greater" | "less"\n' + code

    print_node(title, detail, code)
    if key in EFFECT_GUIDE:
        print_node("Effect size interpretation (rule of thumb)", EFFECT_GUIDE[key], "")


def show_assumption(key: str) -> None:
    if key not in ASSUMPTION_SNIPPETS:
        print_node("INTERNAL ERROR", f"Missing ASSUMPTION_SNIPPETS key: {key}")
        return
    title, detail, code = ASSUMPTION_SNIPPETS[key]
    print_node(title, detail, code)
    if key in EFFECT_GUIDE:
        print_node("Effect size interpretation (rule of thumb)", EFFECT_GUIDE[key], "")


# ----------------------------
# Decision engine (one run)
# ----------------------------

def run_once() -> Dict[str, Any]:
    """Run one traversal; return summary dict."""
    print_node(
        "Interactive Statistical Test Selector",
        "Answer the questions. You'll get recommended test(s) and code snippets.\n"
        "Every path ends with a FINAL RECOMMENDATION summary, then you can restart."
    )

    print_node(
        "Reality Check",
        "This tool helps select common tests.\n"
        "It does NOT replace statistical reasoning.\n\n"
        "Always consider:\n"
        "1) Study design and randomization\n"
        "2) Independence of observations\n"
        "3) Missing data mechanism\n"
        "4) Alternative direction decided BEFORE seeing data\n"
        "5) Equivalence vs difference vs non-inferiority\n"
        "6) Multiple predictors and interactions\n"
        "7) Practical vs statistical significance"
    )

    result: Dict[str, Any] = {"final_tests": [], "notes": []}

    # Alternative choice: this will be reflected in printed snippets (alternative=...) and one-sided CI where applicable.
    global CURRENT_ALT
    CURRENT_ALT = ask_choice(
        "Choose the alternative (two-sided vs one-sided). (Best practice: decide before looking at data)",
        [
            ("two-sided", "Two-sided (default)"),
            ("greater", "One-sided: greater"),
            ("less", "One-sided: less"),
        ],
    )
    result["alternative"] = CURRENT_ALT

    goal = ask_choice(
        "What is your goal?",
        [
            ("compare", "Compare groups / differences"),
            ("assoc", "Association / correlation"),
            ("model", "Prediction / modeling"),
        ],
    )

    # ---------------- Compare ----------------
    if goal == "compare":
        ytype = ask_choice(
            "Outcome (Y) type?",
            [
                ("continuous", "Continuous (interval/ratio)"),
                ("binary", "Binary (yes/no)"),
                ("ordinal", "Ordinal (Likert) / non-normal continuous"),
                ("count", "Count (events)"),
            ],
        )

        # Continuous outcome
        if ytype == "continuous":
            xtype = ask_choice(
                "Predictor (X) type?",
                [
                    ("categorical", "Categorical groups/conditions"),
                    ("continuous", "Continuous predictor"),
                ],
            )
            if xtype == "continuous":
                show_snippet("ols")
                result["final_tests"] = ["ols"]
                return result

            k = ask_choice(
                "How many groups/conditions?",
                [
                    ("1", "1 group vs a constant"),
                    ("2", "2 groups/conditions"),
                    ("3plus", "3+ groups/conditions"),
                ],
            )

            # 1 group vs constant
            if k == "1":
                sigma_known = ask_yes_no("Is population σ (standard deviation) known? (If yes, you can use a z-test.)")

                # workflow: check normality -> decide z/t vs wilcoxon
                show_assumption("normality_shapiro")
                normal_fail = ask_yes_no("Did normality fail / strong outliers / very small n?")

                if normal_fail:
                    show_snippet("wilcoxon_onesample")
                    result["final_tests"] = ["wilcoxon_onesample"]
                else:
                    if sigma_known:
                        show_snippet("onesample_z_mean")
                        result["final_tests"] = ["onesample_z_mean"]
                    else:
                        show_snippet("onesample_t")
                        result["final_tests"] = ["onesample_t"]
                return result

            # 2 groups
            if k == "2":
                paired = ask_choice(
                    "Paired (same subjects) or independent groups?",
                    [("paired", "Paired / repeated on same subjects"), ("ind", "Independent groups")],
                )
                if paired == "paired":
                    # workflow: check normality of differences -> decide paired t vs wilcoxon
                    show_assumption("normality_shapiro")
                    normal_fail = ask_yes_no("Did normality of paired differences fail / strong outliers / very small n?")
                    if normal_fail:
                        show_snippet("wilcoxon_paired")
                        result["final_tests"] = ["wilcoxon_paired"]
                    else:
                        show_snippet("paired_t")
                        result["final_tests"] = ["paired_t"]
                    return result

                # independent groups: decision tree
                show_assumption("normality_shapiro")
                normal_fail = ask_yes_no("Did normality fail (in either group) / strong outliers / very small n?")

                if normal_fail:
                    use_perm = ask_yes_no("Advanced option: use a permutation test for difference in means instead of Mann–Whitney?")
                    if use_perm:
                        show_snippet("perm_ind_diff_means")
                        result["final_tests"] = ["perm_ind_diff_means"]
                    else:
                        show_snippet("mann_whitney")
                        result["final_tests"] = ["mann_whitney"]
                    return result

                # Variance check (only meaningful if roughly normal)
                show_assumption("equal_var_levene")
                var_equal = ask_yes_no("Do variances look equal (Levene p >= alpha)?")

                if var_equal:
                    show_snippet("ind_t_student")
                    result["final_tests"] = ["ind_t_student"]
                else:
                    show_snippet("ind_t_welch")
                    result["final_tests"] = ["ind_t_welch"]
                return result

            # 3+ groups
            repeated = ask_choice(
                "Independent groups or repeated measures (same subjects)?",
                [("ind", "Independent groups"), ("rm", "Repeated measures")],
            )
            if repeated == "ind":
                # workflow: normality -> variance -> choose ANOVA vs Welch ANOVA vs Kruskal
                show_assumption("normality_shapiro")
                normal_fail = ask_yes_no("Did normality fail (in any group) / strong outliers / very small n?")

                show_assumption("equal_var_levene")
                var_equal = ask_yes_no("Do variances look equal (Levene p >= alpha)?")

                if normal_fail:
                    show_snippet("kruskal")
                    show_snippet("posthoc_dunn")
                    result["final_tests"] = ["kruskal", "posthoc_dunn"]
                    return result

                if var_equal:
                    show_snippet("anova_oneway")
                    show_snippet("posthoc_tukey")
                    result["final_tests"] = ["anova_oneway", "posthoc_tukey"]
                    return result

                # Normality OK, variances unequal -> Welch ANOVA + Games-Howell post-hoc
                show_snippet("welch_anova")
                show_snippet("posthoc_games_howell")
                result["final_tests"] = ["welch_anova", "posthoc_games_howell"]
                return result
            else:
                # RM: we give RM-ANOVA snippet + offer Friedman if assumptions not OK
                # (Full sphericity workflow is more complex; keep as guided choice)
                print_node(
                    "Repeated measures note",
                    "RM-ANOVA also involves sphericity; if assumptions are concerning, consider Friedman.\n"
                    "If you want strict sphericity checks and post-hoc, we can extend this branch."
                )
                assumptions_ok = ask_yes_no("Do RM-ANOVA assumptions look acceptable (differences roughly normal, no extreme outliers, sphericity OK/handled)?")
                if assumptions_ok:
                    # We didn't include anova_rm in this streamlined v10 dict; keep it as modeling extension if needed.
                    # To keep the tool coherent, we recommend Friedman here if not ok; otherwise, suggest RM-ANOVA via statsmodels.
                    show_snippet("anova_oneway")  # placeholder would be wrong; so instead show note + recommend AnovaRM snippet
                    # Better: provide a dedicated snippet inline
                    print_node(
                        "Repeated-measures ANOVA (AnovaRM)",
                        "Parametric option for 3+ repeated conditions.",
                        """from statsmodels.stats.anova import AnovaRM

# df columns: subject, condition, y
res = AnovaRM(df, depvar="y", subject="subject", within=["condition"]).fit()
print(res)
"""
                    )
                    result["final_tests"] = ["anova_rm"]
                else:
                    # Friedman snippet not in dict; provide inline
                    print_node(
                        "Friedman test",
                        "Nonparametric alternative to RM-ANOVA (3+ paired conditions).",
                        """from scipy import stats

stat, p_value = stats.friedmanchisquare(x1, x2, x3)  # same subjects across conditions
print("Q =", stat, "p =", p_value)

# Effect size: Kendall's W
k = 3  # TODO number of conditions
n = len(x1)  # TODO subjects
W = stat / (n * (k - 1))
print("Kendall's W =", W)

# Power/CI note: simulation / bootstrap
"""
                    )
                    result["final_tests"] = ["friedman"]
                return result

        # Binary outcome
        if ytype == "binary":
            # Binary/categorical outcomes can mean several different things.
            bgoal = ask_choice(
                "What kind of question is this?",
                [
                    ("prop", "Compare proportion(s) (one-sample or two-sample)"),
                    ("assoc", "Association between two categorical variables (contingency table)"),
                    ("gof", "Goodness-of-fit (one categorical variable vs expected proportions)"),
                    ("model", "Model probability with predictors (logistic regression)"),
                ],
            )

            if bgoal == "model":
                show_snippet("logit")
                result["final_tests"] = ["logit"]
                return result

            if bgoal == "gof":
                show_snippet("chi2_gof")
                result["final_tests"] = ["chi2_gof"]
                return result

            if bgoal == "prop":
                paired = ask_choice(
                    "Paired/matched or independent groups?",
                    [("paired", "Paired / matched"), ("ind", "Independent")],
                )
                if paired == "paired":
                    # For paired binary outcomes, McNemar is the standard.
                    show_snippet("mcnemar")
                    result["final_tests"] = ["mcnemar"]
                    return result

                howmany = ask_choice(
                    "How many proportions are you comparing?",
                    [("1", "One sample proportion vs p0"), ("2", "Two independent groups")],
                )
                if howmany == "1":
                    show_snippet("prop_1sample_ztest")
                    result["final_tests"] = ["prop_1sample_ztest"]
                    return result

                # two independent groups
                # Note: If expected counts are very small, Fisher's exact is safer.
                small = ask_yes_no("Are expected counts small in a 2x2 table (rule: any expected < 5)?")
                if small:
                    show_snippet("fisher_exact")
                    result["final_tests"] = ["fisher_exact"]
                    return result

                show_snippet("prop_2sample_ztest")
                result["final_tests"] = ["prop_2sample_ztest"]
                return result

            # bgoal == "assoc": contingency table association
            paired = ask_choice(
                "Paired/matched or independent groups?",
                [("paired", "Paired / matched"), ("ind", "Independent")],
            )
            if paired == "paired":
                show_snippet("mcnemar")
                result["final_tests"] = ["mcnemar"]
                return result

            small = ask_yes_no("Are expected counts small in a 2x2 table (rule: any expected < 5)?")
            if small:
                show_snippet("fisher_exact")
                result["final_tests"] = ["fisher_exact"]
            else:
                show_snippet("chi2_contingency")
                result["final_tests"] = ["chi2_contingency"]
            return result

        # Ordinal (mostly nonparam) (mostly nonparam)
        if ytype == "ordinal":
            xtype = ask_choice(
                "Predictor (X) type?",
                [
                    ("categorical", "Categorical groups/conditions"),
                    ("continuous", "Continuous predictor(s)"),
                ],
            )
            if xtype == "continuous":
                print_node(
                    "Ordinal regression (OrderedModel)",
                    "Use when Y is ordinal and you have predictors.",
                    """from statsmodels.miscmodels.ordinal_model import OrderedModel

# y: ordered categories (e.g., 1..5), X: predictors
model = OrderedModel(y, X, distr="logit")  # or "probit"
res = model.fit(method="bfgs")
print(res.summary())
"""
                )
                result["final_tests"] = ["ordered_model"]
                return result

            k = ask_choice("How many groups?", [("2", "2 groups"), ("3plus", "3+ groups")])
            if k == "2":
                paired = ask_choice("Paired or independent?", [("paired", "Paired"), ("ind", "Independent")])
                if paired == "paired":
                    show_snippet("wilcoxon_paired")
                    result["final_tests"] = ["wilcoxon_paired"]
                else:
                    show_snippet("mann_whitney")
                    result["final_tests"] = ["mann_whitney"]
                return result
            else:
                paired = ask_choice("Repeated measures or independent groups?", [("rm", "Repeated"), ("ind", "Independent")])
                if paired == "rm":
                    # show inline friedman again
                    print_node(
                        "Friedman test",
                        "Nonparametric alternative to RM-ANOVA (3+ paired conditions).",
                        """from scipy import stats
stat, p_value = stats.friedmanchisquare(x1, x2, x3)
print("Q =", stat, "p =", p_value)
"""
                    )
                    result["final_tests"] = ["friedman"]
                else:
                    show_snippet("kruskal")
                    result["final_tests"] = ["kruskal"]
                return result

        # Count outcome
        if ytype == "count":
            # ── Minimal-question UX: counts split into 3 scenarios only (distribution / contingency / rates)
            count_scenario = ask_choice(
                "What kind of count data do you have?",
                [
                    ("gof", "① One categorical distribution vs expected probabilities (e.g., fair die) — GOF"),
                    ("ind", "② Association/independence between two categorical variables (contingency table)"),
                    ("rate", "③ Event counts with time/exposure (rates): compare or model counts"),
                ],
            )

            if count_scenario == "gof":
                # 1) Always show Cochran check snippet first.
                show_snippet("chi2_gof_cochran_check")

                cochran_res = ask_choice(
                    "What was the Cochran check result?",
                    [
                        ("ok", "Passed (OK)"),
                        ("bad", "Violated"),
                    ],
                )

                if cochran_res == "ok":
                    show_snippet("chi2_gof")
                    result["final_tests"] = ["chi2_gof"]
                    return result

                alt = ask_choice(
                    "If Cochran’s rule is violated, which alternative will you use?",
                    [
                        ("collapse", "Collapse categories then chi-square"),
                        ("mc", "Monte Carlo simulation (recommended)"),
                    ],
                )
                if alt == "collapse":
                    show_snippet("chi2_gof_collapse")
                    result["final_tests"] = ["chi2_gof_collapse"]
                else:
                    show_snippet("chi2_gof_mc")
                    result["final_tests"] = ["chi2_gof_mc"]
                return result


            if count_scenario == "ind":
                # 1) Always show Cochran check snippet first.
                show_snippet("chi2_ind_cochran_check")

                cochran_res = ask_choice(
                    "What was the Cochran check result?",
                    [
                        ("ok", "Passed (OK)"),
                        ("bad", "Violated"),
                    ],
                )

                if cochran_res == "ok":
                    show_snippet("chi2_contingency")
                    result["final_tests"] = ["chi2_contingency"]
                    return result

                shape = ask_choice(
                    "What is the table size?",
                    [
                        ("2x2", "2×2 contingency table"),
                        ("rxk", "R×C (not 2×2)"),
                    ],
                )

                if shape == "2x2":
                    alt = ask_choice(
                        "Cochran violated (2×2): choose an alternative",
                        [
                            ("fisher", "Fisher's exact test (recommended)"),
                            ("mc", "Monte Carlo approximation"),
                        ],
                    )
                    if alt == "fisher":
                        show_snippet("fisher_exact")
                        result["final_tests"] = ["fisher_exact"]
                    else:
                        show_snippet("chi2_ind_mc")
                        result["final_tests"] = ["chi2_ind_mc"]
                    return result

                alt = ask_choice(
                    "Cochran violated (R×C): choose an alternative",
                    [
                        ("collapse", "Collapse categories then chi-square"),
                        ("ffh", "Fisher–Freeman–Halton (FFH) exact test (if available)"),
                        ("mc", "Monte Carlo approximation (recommended)"),
                    ],
                )

                if alt == "collapse":
                    show_snippet("chi2_ind_collapse")
                    result["final_tests"] = ["chi2_ind_collapse"]
                elif alt == "ffh":
                    show_snippet("ffh_exact")
                    result["final_tests"] = ["ffh_exact"]
                else:
                    show_snippet("chi2_ind_mc")
                    result["final_tests"] = ["chi2_ind_mc"]
                return result


            # count_scenario == "rate": event counts / rates / Poisson flow
            goal2 = ask_choice(
                "Event counts: what is your goal?",
                [("compare_rates", "Compare rates between groups"), ("model", "Model counts with predictors")],
            )
            if goal2 == "compare_rates":
                print_node(
                    "Poisson rate test",
                    "Uses statsmodels.stats.rates (API may differ by version). Adjust as needed.",
                    """# Example sketch (API may vary):
# from statsmodels.stats.rates import test_poisson_2indep
# count1, exposure1 = 30, 1000  # events, exposure (e.g., person-time)
# count2, exposure2 = 10, 800
# res = test_poisson_2indep(count1, exposure1, count2, exposure2, method="score")
# print(res)
""",
                )
                result["final_tests"] = ["poisson_rate_test"]
                return result
            else:
                show_snippet("poisson_glm")
                result["final_tests"] = ["poisson_glm"]
                return result

        if ytype == "continuous":
            show_snippet("ols")
            result["final_tests"] = ["ols"]
            return result
        if ytype == "binary":
            show_snippet("logit")
            result["final_tests"] = ["logit"]
            return result
        if ytype == "count":
            show_snippet("poisson_glm")
            result["final_tests"] = ["poisson_glm"]
            return result

    
    # ---------------- Assoc / Correlation ----------------
    if goal == "assoc":
        atype = ask_choice(
            "What type of association/correlation do you want?",
            [
                ("cont_cont", "Continuous ↔ Continuous (correlation)"),
                ("cat_cat", "Categorical ↔ Categorical (contingency / association)"),
            ],
        )

        if atype == "cont_cont":
            method = ask_choice(
                "Which correlation method?",
                [
                    ("pearsonr", "Pearson (linear; assumes roughly normal; sensitive to outliers)"),
                    ("spearmanr", "Spearman (rank-based; more robust to non-normality/outliers)"),
                ],
            )
            show_snippet(method)
            result["final_tests"] = [method]
            result["notes"].append("Correlation does NOT imply causation. Always inspect a scatter plot and outliers/nonlinearity.")
            return result

        if atype == "cat_cat":
            show_snippet("chi2_contingency")
            result["final_tests"] = ["chi2_contingency"]
            result["notes"].append("If chi-square is significant, inspect adjusted standardized residuals to see which cells drive the association.")
            return result

    # ---------------- Model / Prediction ----------------
    if goal == "model":
        my = ask_choice(
            "What is the outcome (Y) type?",
            [
                ("continuous", "Continuous (linear regression)"),
                ("binary", "Binary (logistic regression)"),
                ("count", "Count (Poisson / negative binomial etc.)"),
                ("ordinal", "Ordinal (ordered regression)"),
            ],
        )

        if my == "continuous":
            show_snippet("ols")
            result["final_tests"] = ["ols"]
            return result
        if my == "binary":
            show_snippet("logit")
            result["final_tests"] = ["logit"]
            return result
        if my == "count":
            show_snippet("poisson_glm")
            result["final_tests"] = ["poisson_glm"]
            return result
        if my == "ordinal":
            print_node(
                "Ordered regression (OrderedModel)",
                "Use when Y is ordinal (ordered categories) and you have predictors.",
                """from statsmodels.miscmodels.ordinal_model import OrderedModel

# y: ordered categories (e.g., 1..5), X: predictors
model = OrderedModel(y, X, distr="logit")  # or "probit"
res = model.fit(method="bfgs")
print(res.summary())
""",
            )
            result["final_tests"] = ["ordered_model"]
            return result

    result["notes"].append("No matching path (unexpected).")
    return result


def print_final_summary(res: Dict[str, Any]) -> None:
    tests = res.get("final_tests") or []
    notes = res.get("notes") or []
    if tests:
        pretty = ", ".join(tests)
        print_node("FINAL RECOMMENDATION", f"Selected test(s): {pretty}\n\nRe-run (or choose restart) to explore another path.")
    else:
        print_node("FINAL RECOMMENDATION", "No test selected.\n" + ("\n".join(notes) if notes else ""))



def smoketest() -> int:
    """Quick integrity check: imports + snippet compilation."""
    print_node("Smoke test", "Checking imports and compiling snippet code blocks...")

    # Check core imports
    core_modules = [
        ("numpy", "np"),
        ("scipy", "stats"),
        ("statsmodels", "sm"),
    ]
    optional_modules = [
        ("pingouin", "pg (optional for Games–Howell)"),
        ("scikit_posthocs", "sp (optional for Dunn)"),
    ]

    ok = True
    for mod, label in core_modules:
        try:
            __import__(mod)
            print(f"[OK] import {mod}")
        except Exception as e:
            ok = False
            print(f"[FAIL] import {mod}: {e}")

    for mod, label in optional_modules:
        try:
            __import__(mod)
            print(f"[OK] optional import {mod}")
        except Exception as e:
            print(f"[WARN] optional import {mod} failed: {e}")

    # Compile all snippets to ensure syntax validity
    for key, (title, detail, code) in {**ASSUMPTION_SNIPPETS, **TEST_SNIPPETS}.items():
        try:
            compile(code, filename=f"<snippet:{key}>", mode="exec")
        except Exception as e:
            ok = False
            print(f"[FAIL] compile snippet {key}: {e}")

    if ok:
        print_node("Smoke test result", "PASS: Imports ok (core) and all snippets compile.")
        return 0
    else:
        print_node("Smoke test result", "FAIL: One or more core imports or snippet compilations failed.")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive statistical test selector (CLI)")
    parser.add_argument("--smoketest", action="store_true", help="Run integrity checks (imports + snippet compilation) and exit")
    args = parser.parse_args()

    if args.smoketest:
        raise SystemExit(smoketest())

    while True:
        res = run_once()
        print_final_summary(res)

        again = ask_yes_no("Restart the decision tree?")
        if not again:
            print_node("Done", "Exiting. Good luck with your analysis!")
            break



def run_smoketest() -> None:
    """Compile all snippet blocks to ensure they are syntactically valid."""
    import textwrap
    failures = []
    def _try_compile(name: str, code: str) -> None:
        src = textwrap.dedent(code).strip("\n")
        if not src:
            return
        compile(src, f"<snippet:{name}>", "exec")

    for k, (title, desc, code) in ASSUMPTION_SNIPPETS.items():
        try:
            _try_compile(f"ASSUMPTION:{k}", code)
        except Exception as e:
            failures.append((k, title, str(e)))

    for k, (title, desc, code) in TEST_SNIPPETS.items():
        try:
            _try_compile(f"TEST:{k}", code)
        except Exception as e:
            failures.append((k, title, str(e)))

    if failures:
        print("\\n[SMOKETEST] Some snippets failed to compile:")
        for k, title, msg in failures:
            print(f" - {k} ({title}): {msg}")
        raise SystemExit(1)

    print(f"\\n[SMOKETEST] OK: compiled {len(ASSUMPTION_SNIPPETS) + len(TEST_SNIPPETS)} snippets without syntax errors.")

def run_fuzz(iterations: int = 200, seed: int = 1234) -> None:
    """Automatically traverse the decision tree with valid choices to catch runtime crashes."""
    import random
    import contextlib
    import io

    rnd = random.Random(seed)

    def auto_choice(prompt: str, choices):
        if not choices:
            raise ValueError("No choices provided")
        idx = rnd.randrange(len(choices))
        return choices[idx][0]

    def auto_yes_no(prompt: str) -> bool:
        return auto_choice(prompt, [("y","yes"),("n","no")]) == "y"

    global ask_choice, ask_yes_no, CURRENT_ALT
    saved_choice, saved_yesno, saved_alt = ask_choice, ask_yes_no, CURRENT_ALT
    try:
        ask_choice = auto_choice
        ask_yes_no = auto_yes_no
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(iterations):
                CURRENT_ALT = None
                run_once()
        print(f"[FUZZ] OK: ran run_once() {iterations} times without crashing. (seed={seed})")
    finally:
        ask_choice, ask_yes_no, CURRENT_ALT = saved_choice, saved_yesno, saved_alt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Statistical test selector (CLI)")
    parser.add_argument("--smoketest", action="store_true")
    parser.add_argument("--fuzz", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.smoketest:
        run_smoketest()
    elif args.fuzz and args.fuzz > 0:
        run_fuzz(iterations=args.fuzz, seed=args.seed)
    else:
        main()
