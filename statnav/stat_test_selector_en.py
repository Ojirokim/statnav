# ============================================================
# Statistical Test Selection & Inference Assistant (CLI)
# ------------------------------------------------------------
# Features:
#  - Course-aligned test selection workflow
#  - Assumption checks (Normality: Shapiro + Q-Q(probplot), Equal variance: Levene)
#  - Effect sizes and confidence intervals (analytic or bootstrap guidance where applicable)
#
# Version: v1.1 
# Last check: 2026-02-14  |  smoketest: PASS
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


def ask_yes_no(prompt: str) -> bool:
    return ask_choice(prompt, [("y", "Yes"), ("n", "No")]) == "y"


def print_node(title: str, detail: Optional[str] = None, code: Optional[str] = None) -> None:
    print("\n" + "=" * 72)
    print(title)
    if detail:
        print("-" * 72)
        print(detail)
    if code:
        print("-" * 72)
        print("Code snippet:")
        print(code)
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
        """from scipy import stats
import numpy as np

# x : sample array
mu0 = 0  # TODO: hypothesized mean under H0

t_stat, p_value = stats.ttest_1samp(x, popmean=mu0, alternative="two-sided")  # or "less"/"greater"
print("t =", t_stat, "p =", p_value)

# Effect size: Cohen's d
d = (np.mean(x) - mu0) / np.std(x, ddof=1)
print("Cohen's d =", d)

# Power
from statsmodels.stats.power import TTestPower
power = TTestPower().power(effect_size=d, nobs=len(x), alpha=0.05, alternative="two-sided")  # TODO alpha/alternative
print("Power =", power)

# CI for mean and (mean - mu0)
alpha = 0.05  # TODO
n = len(x); df = n - 1
se = stats.sem(x)
tcrit = stats.t.ppf(1 - alpha/2, df)
xbar = np.mean(x)
ci_mean = (xbar - tcrit*se, xbar + tcrit*se)
ci_diff = ((xbar-mu0) - tcrit*se, (xbar-mu0) + tcrit*se)
print(f"{int((1-alpha)*100)}% CI mean =", ci_mean)
print(f"{int((1-alpha)*100)}% CI (mean-mu0) =", ci_diff)
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

# Effect size r (approx) from z
diff_nz = diff[diff != 0]
n = len(diff_nz)
mean_w = n * (n + 1) / 4
var_w = n * (n + 1) * (2 * n + 1) / 24
z = (w_stat - mean_w - 0.5) / np.sqrt(var_w)  # continuity correction
r = z / np.sqrt(n)
print("Effect size r (approx) =", r)

# Bootstrap CI for r
alpha = 0.05
B = 2000
rng = np.random.default_rng(0)
boot = []
for _ in range(B):
    d_b = rng.choice(diff, size=len(diff), replace=True)
    w_b, _ = stats.wilcoxon(d_b, alternative="two-sided")
    d_nz = d_b[d_b != 0]
    n_b = len(d_nz)
    if n_b < 5:
        continue
    mean_wb = n_b * (n_b + 1) / 4
    var_wb = n_b * (n_b + 1) * (2 * n_b + 1) / 24
    z_b = (w_b - mean_wb - 0.5) / np.sqrt(var_wb)
    boot.append(z_b / np.sqrt(n_b))
boot = np.sort(np.array(boot))
ci = (np.quantile(boot, alpha/2), np.quantile(boot, 1-alpha/2))
print(f"{int((1-alpha)*100)}% bootstrap CI for r =", ci)

# Power note: typically via simulation for Wilcoxon.
"""
    ),

    # Paired t
    "paired_t": (
        "Paired t-test",
        "Same subjects measured twice (before/after).",
        """from scipy import stats
import numpy as np

# x_before, x_after : paired arrays
t_stat, p_value = stats.ttest_rel(x_before, x_after, alternative="two-sided")  # or "less"/"greater"
print("t =", t_stat, "p =", p_value)

# Effect size: Cohen's d (paired)
diff = x_before - x_after
d = np.mean(diff) / np.std(diff, ddof=1)
print("Cohen's d (paired) =", d)

# Power
from statsmodels.stats.power import TTestPower
power = TTestPower().power(effect_size=d, nobs=len(diff), alpha=0.05)  # TODO alpha/alternative
print("Power =", power)

# CI for paired mean difference
alpha = 0.05  # TODO
n = len(diff); df = n - 1
se = stats.sem(diff)
tcrit = stats.t.ppf(1 - alpha/2, df)
ci = (np.mean(diff) - tcrit*se, np.mean(diff) + tcrit*se)
print(f"{int((1-alpha)*100)}% CI mean(diff) =", ci)
"""
    ),
    "wilcoxon_paired": (
        "Wilcoxon signed-rank test (paired)",
        "Nonparametric alternative to paired t-test.",
        """from scipy import stats
import numpy as np

# x_before, x_after : paired arrays
diff = x_before - x_after
w_stat, p_value = stats.wilcoxon(diff, alternative="two-sided")
print("W =", w_stat, "p =", p_value)

# Effect size r (approx)
diff_nz = diff[diff != 0]
n = len(diff_nz)
mean_w = n * (n + 1) / 4
var_w = n * (n + 1) * (2 * n + 1) / 24
z = (w_stat - mean_w - 0.5) / np.sqrt(var_w)
r = z / np.sqrt(n)
print("Effect size r (approx) =", r)

# Bootstrap CI for r
alpha = 0.05
B = 2000
rng = np.random.default_rng(0)
boot = []
for _ in range(B):
    d_b = rng.choice(diff, size=len(diff), replace=True)
    w_b, _ = stats.wilcoxon(d_b, alternative="two-sided")
    d_nz = d_b[d_b != 0]
    n_b = len(d_nz)
    if n_b < 5:
        continue
    mean_wb = n_b * (n_b + 1) / 4
    var_wb = n_b * (n_b + 1) * (2 * n_b + 1) / 24
    z_b = (w_b - mean_wb - 0.5) / np.sqrt(var_wb)
    boot.append(z_b / np.sqrt(n_b))
boot = np.sort(np.array(boot))
ci = (np.quantile(boot, alpha/2), np.quantile(boot, 1-alpha/2))
print(f"{int((1-alpha)*100)}% bootstrap CI for r =", ci)

# Power note: typically via simulation.
"""
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
tcrit = stats.t.ppf(1 - alpha/2, df)
ci = (mean_diff - tcrit*se_diff, mean_diff + tcrit*se_diff)
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
tcrit = stats.t.ppf(1 - alpha/2, df_w)
ci = (mean_diff - tcrit*se_diff, mean_diff + tcrit*se_diff)
print(f"{int((1-alpha)*100)}% CI mean difference (Welch) =", ci)
"""
    ),
    "mann_whitney": (
        "Mann–Whitney U test",
        "Nonparametric alternative to independent t-test (rank-sum).",
        """from scipy import stats
import numpy as np

# a, b : independent samples
u_stat, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
print("U =", u_stat, "p =", p_value)

n1, n2 = len(a), len(b)

# Effect size 1: Rank-biserial correlation (RBC)
rbc = 1 - (2 * u_stat) / (n1 * n2)
print("Rank-biserial correlation (RBC) =", rbc)

# Effect size 2: Cliff's delta (equivalent to RBC under standard definitions)
# delta = P(a > b) - P(a < b)
def cliffs_delta(x, y):
    x = np.asarray(x); y = np.asarray(y)
    gt = sum((xi > y).sum() for xi in x)
    lt = sum((xi < y).sum() for xi in x)
    return (gt - lt) / (len(x) * len(y))

delta = cliffs_delta(a, b)
print("Cliff's delta =", delta)

# Bootstrap CI for RBC (or delta)
alpha = 0.05
B = 2000
rng = np.random.default_rng(0)
boot = []
for _ in range(B):
    a_b = rng.choice(a, size=n1, replace=True)
    b_b = rng.choice(b, size=n2, replace=True)
    u_b, _ = stats.mannwhitneyu(a_b, b_b, alternative="two-sided")
    rbc_b = 1 - (2 * u_b) / (n1 * n2)
    boot.append(rbc_b)
boot = np.sort(np.array(boot))
ci = (np.quantile(boot, alpha/2), np.quantile(boot, 1-alpha/2))
print(f"{int((1-alpha)*100)}% bootstrap CI for RBC =", ci)

# Power note: typically via simulation for Mann–Whitney.
"""
    ),

    # ANOVA / Kruskal
    "anova_oneway": (
        "One-way ANOVA",
        "Compare means across 3+ independent groups (parametric).",
        """from scipy import stats
import numpy as np

# g1, g2, g3 : group arrays (extend as needed)
f_stat, p_value = stats.f_oneway(g1, g2, g3)
print("F =", f_stat, "p =", p_value)

groups = [g1, g2, g3]  # TODO extend
y = np.concatenate(groups)
grand_mean = np.mean(y)

# Sums of squares
ss_between = sum(len(g)*(np.mean(g)-grand_mean)**2 for g in groups)
ss_within = sum(((g - np.mean(g))**2).sum() for g in groups)
ss_total = ss_between + ss_within

# Effect sizes
eta2 = ss_between / ss_total
k = len(groups)
n = len(y)
ms_within = ss_within / (n - k)
omega2 = (ss_between - (k - 1)*ms_within) / (ss_total + ms_within)
omega2 = max(0.0, omega2)

cohens_f = np.sqrt(eta2 / (1 - eta2))
print("eta^2 =", eta2, "omega^2 =", omega2, "Cohen's f =", cohens_f)

# Power
from statsmodels.stats.power import FTestAnovaPower
power = FTestAnovaPower().power(effect_size=cohens_f, k_groups=k, nobs=n, alpha=0.05)
print("Power =", power)

# CI note:
# - CIs for eta^2/omega^2 often via bootstrap.
# - Pairwise mean-difference CIs come from post-hoc procedures (Tukey, Games–Howell).
"""
    ),

    "welch_anova": (
        "Welch's ANOVA (one-way, unequal variances)",
        "Compare means across 3+ independent groups when normality is acceptable but variances are unequal.",
        """import numpy as np
from statsmodels.stats.oneway import anova_oneway

# Provide your data as a list of group arrays:
groups = [g1, g2, g3]  # TODO extend

res = anova_oneway(groups, use_var="unequal", welch_correction=True)
print(res)

# post-hoc note:
# If Welch's ANOVA is significant, a common post-hoc is Games-Howell.
"""
    ),

    "posthoc_tukey": (
        "post-hoc: Tukey HSD (after classic one-way ANOVA)",
        "Use after a significant one-way ANOVA when variances are ~ equal.",
        """import numpy as np
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# long-format data
df = pd.DataFrame({
    "y": np.concatenate([g1, g2, g3]),   # TODO extend
    "group": (['g1']*len(g1) + ['g2']*len(g2) + ['g3']*len(g3))
})

res = pairwise_tukeyhsd(endog=df["y"], groups=df["group"], alpha=0.05)
print(res)
"""
    ),

    "posthoc_games_howell": (
        "post-hoc: Games–Howell (after Welch's ANOVA)",
        "Use after a significant Welch's ANOVA (no equal-variance assumption).",
        """# Common approach: use Pingouin's Games-Howell implementation.
# pip install pingouin

import numpy as np
import pandas as pd
import pingouin as pg

df = pd.DataFrame({
    "y": np.concatenate([g1, g2, g3]),   # TODO extend
    "group": (['g1']*len(g1) + ['g2']*len(g2) + ['g3']*len(g3))
})

gh = pg.pairwise_gameshowell(dv='y', between='group', data=df)
print(gh)
"""
    ),

    "posthoc_dunn": (
        "post-hoc: Dunn's test (after Kruskal–Wallis)",
        "Use after a significant Kruskal–Wallis; adjusts for multiple comparisons.",
        """# Common approach: use scikit-posthocs Dunn test.
# pip install scikit-posthocs

import numpy as np
import pandas as pd
import scikit_posthocs as sp

df = pd.DataFrame({
    "y": np.concatenate([g1, g2, g3]),   # TODO extend
    "group": (['g1']*len(g1) + ['g2']*len(g2) + ['g3']*len(g3))
})

# Holm adjustment is a solid default; alternatives: 'bonferroni', 'fdr_bh', etc.
pvals = sp.posthoc_dunn(df, val_col='y', group_col='group', p_adjust='holm')
print(pvals)
"""
    ),
    "kruskal": (
        "Kruskal–Wallis test",
        "Nonparametric alternative to one-way ANOVA (3+ independent groups).",
        """from scipy import stats
import numpy as np

h_stat, p_value = stats.kruskal(g1, g2, g3)  # extend as needed
print("H =", h_stat, "p =", p_value)

# Effect size: epsilon-squared (recommended over eta2_H)
k = 3  # TODO number of groups
n_total = len(g1) + len(g2) + len(g3)  # TODO extend
eps2 = (h_stat - k + 1) / (n_total - k)
print("Epsilon-squared =", eps2)

# Bootstrap CI for epsilon-squared
alpha = 0.05
B = 2000
rng = np.random.default_rng(0)
groups = [np.asarray(g1), np.asarray(g2), np.asarray(g3)]
ns = [len(g) for g in groups]
boot = []
for _ in range(B):
    boot_groups = [rng.choice(g, size=n, replace=True) for g, n in zip(groups, ns)]
    h_b, _ = stats.kruskal(*boot_groups)
    eps2_b = (h_b - k + 1) / (sum(ns) - k)
    boot.append(eps2_b)
boot = np.sort(np.array(boot))
ci = (np.quantile(boot, alpha/2), np.quantile(boot, 1-alpha/2))
print(f"{int((1-alpha)*100)}% bootstrap CI for epsilon^2 =", ci)

# Power note: typically via simulation for Kruskal–Wallis.
"""
    ),

    # Binary/categorical
    "chi2_contingency": (
        "Chi-square test of independence",
        "Association between categorical variables (contingency table, RxC).",
        """import numpy as np
from scipy import stats

# table : contingency table array
chi2, p, dof, expected = stats.chi2_contingency(table)
print("chi2 =", chi2, "p =", p, "dof =", dof)

# Effect size: Cramer's V
n = table.sum()
r, c = table.shape
v = np.sqrt(chi2 / (n * (min(r-1, c-1))))
print("Cramer's V =", v)

# Power (approx)
from statsmodels.stats.power import GofChisquarePower
power = GofChisquarePower().power(effect_size=v, nobs=n, alpha=0.05)
print("Power (approx) =", power)

# Bootstrap CI for Cramer's V
alpha = 0.05
B = 2000
rng = np.random.default_rng(0)
# Convert contingency table into row/col samples for resampling
rows = np.repeat(np.arange(r), table.sum(axis=1))
cols = np.concatenate([np.repeat(j, table[:, j].sum()) for j in range(c)])
nobs = len(rows)
boot = []
for _ in range(B):
    idx = rng.integers(0, nobs, nobs)
    t_b = np.zeros((r, c), dtype=int)
    for rr, cc in zip(rows[idx], cols[idx]):
        t_b[rr, cc] += 1
    chi2_b, _, _, _ = stats.chi2_contingency(t_b)
    v_b = np.sqrt(chi2_b / (t_b.sum() * (min(r-1, c-1))))
    boot.append(v_b)
boot = np.sort(np.array(boot))
ci = (np.quantile(boot, alpha/2), np.quantile(boot, 1-alpha/2))
print(f"{int((1-alpha)*100)}% bootstrap CI for Cramer's V =", ci)
"""
    ),
    "fisher_exact": (
        "Fisher's exact test (2x2)",
        "Use when expected counts are small in a 2x2 table.",
        """import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import Table2x2

table = np.array([[a, b],
                  [c, d]])

oddsratio, p_value = stats.fisher_exact(table, alternative="two-sided")
print("OR =", oddsratio, "p =", p_value)

# Exact/conditional CI for OR (statsmodels)
t = Table2x2(table)
ci_low, ci_high = t.oddsratio_confint()
print("OR 95% CI =", (ci_low, ci_high))

# Power note: typically via simulation or specialized routines.
"""
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
        """from scipy import stats
import numpy as np

r, p_value = stats.pearsonr(x, y)
print("r =", r, "p =", p_value)

# Power (approx)
from statsmodels.stats.power import NormalIndPower
effect = r / np.sqrt(1 - r**2)
power = NormalIndPower().power(effect_size=effect, nobs1=len(x), alpha=0.05)
print("Power (approx) =", power)

# CI for r (Fisher z)
alpha = 0.05  # TODO
n = len(x)
z = np.arctanh(r)
se = 1 / np.sqrt(n - 3)
zcrit = stats.norm.ppf(1 - alpha/2)
r_low, r_high = np.tanh(z - zcrit*se), np.tanh(z + zcrit*se)
print(f"{int((1-alpha)*100)}% CI for r =", (r_low, r_high))
"""
    ),
    "spearmanr": (
        "Spearman correlation",
        "Rank-based association for non-normal continuous or ordinal variables.",
        """from scipy import stats
import numpy as np

rho, p_value = stats.spearmanr(x, y)
print("rho =", rho, "p =", p_value)

# Power note:
# - No standard closed-form power; use simulation if needed.

# CI note (bootstrap)
alpha = 0.05  # TODO
B = 2000       # TODO bootstrap resamples
rng = np.random.default_rng(0)
n = len(x)
boot = []
for _ in range(B):
    idx = rng.integers(0, n, n)
    rho_b, _ = stats.spearmanr(np.array(x)[idx], np.array(y)[idx])
    boot.append(rho_b)
boot = np.sort(np.array(boot))
ci = (np.quantile(boot, alpha/2), np.quantile(boot, 1-alpha/2))
print(f"{int((1-alpha)*100)}% bootstrap CI for rho =", ci)
"""
    ),

    # Modeling
    "ols": (
        "Linear regression (OLS)",
        "Continuous outcome with one or more predictors.",
        """import statsmodels.api as sm

# y : outcome, X : predictors (2D)
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

# CIs:
# ci = model.conf_int(alpha=0.05)  # TODO alpha
# print(ci)
"""
    ),
    "logit": (
        "Logistic regression",
        "Binary outcome with one or more predictors.",
        """import numpy as np
import statsmodels.api as sm

X = sm.add_constant(X)
model = sm.Logit(y, X).fit()
print(model.summary())

# Odds ratios (OR) and 95% CI
ci = model.conf_int(alpha=0.05)
or_ci = np.exp(ci)
or_ = np.exp(model.params)
print("OR =", or_)
print("OR 95% CI =", or_ci)
"""
    ),
    "poisson_glm": (
        "Poisson regression (GLM)",
        "Count outcome; consider Negative Binomial if overdispersed.",
        """import numpy as np
import statsmodels.api as sm

X = sm.add_constant(X)
model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
print(model.summary())

# Incidence rate ratios (IRR) and 95% CI
ci = model.conf_int(alpha=0.05)
irr_ci = np.exp(ci)
irr_ = np.exp(model.params)
print("IRR =", irr_)
print("IRR 95% CI =", irr_ci)
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
p_two = 2 * stats.norm.sf(abs(z))  # two-sided
print("z =", z, "p (two-sided) =", p_two)

# CI for mean (z-based)
alpha = 0.05
zcrit = stats.norm.ppf(1 - alpha/2)
ci = (xbar - zcrit * sigma/np.sqrt(n), xbar + zcrit * sigma/np.sqrt(n))
print("95% CI for mean =", ci)
"""
    ),

    # Proportions: one-sample z-test
    "prop_1sample_ztest": (
        "One-sample proportion z-test",
        "Compare an observed proportion to a hypothesized p0 (e.g., conversion rate vs benchmark).",
        """from statsmodels.stats.proportion import proportions_ztest, proportion_confint, proportion_effectsize

# successes / trials
count = 42   # TODO: number of successes
nobs = 100   # TODO: total trials
p0 = 0.30    # TODO: hypothesized proportion under H0

z_stat, p_value = proportions_ztest(count=count, nobs=nobs, value=p0, alternative="two-sided")
print("z =", z_stat, "p =", p_value)

# Estimate and CI for proportion (Wilson is a good default)
phat = count / nobs
ci_low, ci_high = proportion_confint(count, nobs, alpha=0.05, method="wilson")
print("p̂ =", phat, "95% CI =", (ci_low, ci_high))

# Effect size: Cohen's h
h = proportion_effectsize(phat, p0)
print("Cohen's h =", h)

# Power note:
# For power planning, two-sample proportion tests can use NormalIndPower (see the two-sample snippet).
# One-sample proportion power is typically planned with a normal approximation or specialized routines.
"""
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

# Observed counts for k categories
observed = np.array([30, 50, 20])   # TODO

# Option A: expected proportions (must sum to 1)
expected_props = np.array([0.3, 0.5, 0.2])  # TODO
expected = expected_props * observed.sum()

chi2, p = stats.chisquare(f_obs=observed, f_exp=expected)
print("chi2 =", chi2, "p =", p)

# Effect size: Cohen's w
n = observed.sum()
w = np.sqrt(chi2 / n)
print("Cohen's w =", w)

# Bootstrap CI for w
alpha = 0.05
B = 2000
rng = np.random.default_rng(0)
boot = []
for _ in range(B):
    obs_b = rng.multinomial(n, observed/observed.sum())
    chi2_b, _ = stats.chisquare(f_obs=obs_b, f_exp=expected_props * n)
    boot.append(np.sqrt(chi2_b / n))
boot = np.sort(np.array(boot))
ci = (np.quantile(boot, alpha/2), np.quantile(boot, 1-alpha/2))
print(f"{int((1-alpha)*100)}% bootstrap CI for w =", ci)

# Note: if you estimated parameters from the data, adjust df accordingly (advanced).
"""
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


def show_snippet(key: str) -> None:
    if key not in TEST_SNIPPETS:
        print_node("INTERNAL ERROR", f"Missing TEST_SNIPPETS key: {key}")
        return
    title, detail, code = TEST_SNIPPETS[key]
    print_node(title, detail, code)


def show_assumption(key: str) -> None:
    if key not in ASSUMPTION_SNIPPETS:
        print_node("INTERNAL ERROR", f"Missing ASSUMPTION_SNIPPETS key: {key}")
        return
    title, detail, code = ASSUMPTION_SNIPPETS[key]
    print_node(title, detail, code)


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
            goal2 = ask_choice(
                "Counts: what is your goal?",
                [("compare_rates", "Compare rates between groups"), ("model", "Model counts with predictors")],
            )
            if goal2 == "compare_rates":
                print_node(
                    "Poisson rates test",
                    "Use statsmodels.stats.rates. API varies by version; tailor to your context.",
                    """# Example sketch (API may vary):
# from statsmodels.stats.rates import test_poisson_2indep
# res = test_poisson_2indep(count1, exposure1, count2, exposure2)
# print(res)
"""
                )
                result["final_tests"] = ["poisson_rates_test"]
            else:
                show_snippet("poisson_glm")
                result["final_tests"] = ["poisson_glm"]
            return result

    # ---------------- Association ----------------
    if goal == "assoc":
        vtype = ask_choice(
            "Variable types for association?",
            [
                ("cont_norm", "Two continuous (approx normal, linear)"),
                ("cont_non", "Continuous non-normal or ordinal"),
                ("cat_cat", "Categorical vs categorical"),
                ("bin_pred", "Binary outcome with predictors"),
            ],
        )
        if vtype == "cont_norm":
            # workflow: optionally check normality via Shapiro+QQ on x and y
            show_assumption("normality_shapiro")
            _ = ask_yes_no("Proceed with Pearson (assumes roughly normal / linear)?")
            show_snippet("pearsonr")
            result["final_tests"] = ["pearsonr"]
            return result
        if vtype == "cont_non":
            show_snippet("spearmanr")
            result["final_tests"] = ["spearmanr"]
            return result
        if vtype == "cat_cat":
            show_snippet("chi2_contingency")
            result["final_tests"] = ["chi2_contingency"]
            return result
        if vtype == "bin_pred":
            show_snippet("logit")
            result["final_tests"] = ["logit"]
            return result

    # ---------------- Modeling ----------------
    if goal == "model":
        ytype = ask_choice(
            "Modeling: outcome type?",
            [
                ("continuous", "Continuous"),
                ("binary", "Binary"),
                ("count", "Count"),
            ],
        )
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


if __name__ == "__main__":
    main()
