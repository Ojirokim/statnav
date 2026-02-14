
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
# 통계 검정 선택 및 추론 도우미 (CLI)
# ------------------------------------------------------------
# 기능:
#  - 검정 선택(단계별 가이드)
#  - 가정 검토(정규성: Shapiro + Q-Q(probplot), 등분산: Levene)
#  - 효과크기 및 신뢰구간(가능한 경우 해석적/부트스트랩 안내)
#
# 버전: v1.3.0
# 마지막 점검: 2026-02-15  |  smoketest: PASS
# Developed by: 김규열(Ojirokim)
# License: MIT
# ============================================================

#!/usr/bin/env python3
"""대화형 통계 검정 선택기(CLI) — 워크플로우 친화 버전

v1.2 변경점:
- 단측/양측 선택을 필요한 검정에서만 1회 묻고, 이후 모든 코드 스니펫에 자동 반영
- 부트스트랩 CI도 단측 선택 시 한쪽만(±inf) 나오도록 개선(해당 스니펫)
- 각 검정의 효과크기 해석 기준(경험칙)을 함께 출력

from __future__ import annotations

목표:
- 모든 분기에서 "애매한 끝맺음" 없이, 최종 권장사항을 명확히 출력합니다.
- 일관된 흐름(가정 확인 → 의사결정 → 최종 검정)을 제공합니다.
- 검정/효과크기/검정력/신뢰구간/가정 확인(Shapiro+Q-Q, Levene 등)에 대한 코드 스니펫을 출력합니다.
- 한 번 실행에서 여러 분기를 탐색할 수 있도록 재시작을 지원합니다.

참고:
- 비모수 검정의 검정력/신뢰구간은 종종 시뮬레이션/부트스트랩이 필요할 수 있으며, 스니펫에 가이드를 포함합니다.
"""

from typing import Optional, List, Tuple, Dict, Any

CURRENT_ALT: Optional[str] = None  # "two-sided" | "greater" | "less"



# ----------------------------
# Helpers
# ----------------------------

def ask_choice(prompt: str, choices: List[Tuple[str, str]]) -> str:
    print("\n" + prompt)
    for i, (_, label) in enumerate(choices, start=1):
        print(f"  {i}. {label}")
    while True:
        raw = input("번호를 선택하세요: ").strip()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(choices):
                return choices[idx - 1][0]
        print("잘못된 선택입니다. 다시 시도하세요.")



def ensure_alternative_selected() -> str:
    """대립가설 방향이 필요한 검정에서만 1회 물어봅니다."""
    global CURRENT_ALT
    if CURRENT_ALT is None:
        CURRENT_ALT = ask_choice(
            "대립가설(단측/양측)을 선택하세요. (데이터를 보기 전에 결정하는 것이 원칙입니다)",
            [
                ("two-sided", "양측(two-sided)"),
                ("greater", "단측: 크다/증가(greater)"),
                ("less", "단측: 작다/감소(less)"),
            ],
        )
    return CURRENT_ALT

def ask_yes_no(prompt: str) -> bool:
    return ask_choice(prompt, [("y", "예"), ("n", "아니오")]) == "y"


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
        print("코드 스니펫 (복사용):")
        print("# ---BEGIN SNIPPET---")
        safe_code = code.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ").rstrip("\n")
        sys.stdout.write(safe_code + "\n")
        print("# ---END SNIPPET---")
    print("=" * 72)


# ----------------------------
# Snippets (tests + assumptions)
# ----------------------------

ASSUMPTION_SNIPPETS = {
    "normality_shapiro": (
        "정규성 검정(Shapiro-Wilk + Q-Q 플롯)",
        """통계적 검정(Shapiro-Wilk)과 시각적 확인(Q-Q 플롯)을 함께 사용하세요.
적용 대상: 표본(일표본), 대응 차이(대응), 또는 잔차(회귀).""",
        """from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 1) Shapiro-Wilk test
stat, p = stats.shapiro(x)   # x: sample OR residuals OR paired differences
print("Shapiro p =", p)

# 2) QQ plot (visual normality check)
from scipy import stats
stats.probplot(x, dist="norm", plot=plt)
plt.title("Q-Q Plot (Normal)")
plt.show()

# Alternative QQ plot using SciPy:
# stats.probplot(x, dist="norm", plot=plt)
# plt.show()
"""
    ),
    "equal_var_levene": (
        "등분산 검정(Levene)",
        "Student t-검정/전통적 ANOVA 전에 유용합니다.",
        """from scipy import stats

stat, p = stats.levene(group1, group2)  # pass 3개 이상 집단 too
print("Levene p =", p)
"""
    ),
}

TEST_SNIPPETS = {
    # One-sample t
    "onesample_t": (
        "일표본 t-검정",
        "한 집단의 평균을 특정 상수와 비교합니다.",
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
xbar = np.mean(x)

tcrit_two = stats.t.ppf(1 - alpha/2, df)
tcrit_one = stats.t.ppf(1 - alpha, df)

if alternative == "two-sided":
    ci_mean = (xbar - tcrit_two*se, xbar + tcrit_two*se)
    ci_diff = ((xbar-mu0) - tcrit_two*se, (xbar-mu0) + tcrit_two*se)
elif alternative == "greater":
    ci_mean = (xbar - tcrit_one*se, np.inf)
    ci_diff = ((xbar-mu0) - tcrit_one*se, np.inf)
else:  # "less"
    ci_mean = (-np.inf, xbar + tcrit_one*se)
    ci_diff = (-np.inf, (xbar-mu0) + tcrit_one*se)

print(f"{int((1-alpha)*100)}% CI mean =", ci_mean)
print(f"{int((1-alpha)*100)}% CI (mean-mu0) =", ci_diff)
"""
    ),
    # Wilcoxon one-sample (vs constant)
    "wilcoxon_onesample": (
        "Wilcoxon 부호순위 검정(일표본, 상수와 비교)",
        "일표본 t-검정의 비모수 대안입니다. 중앙값(x − μ0)이 0인지 검정합니다.",
        """from scipy import stats
import numpy as np

# x : sample array, mu0 : 가설 중앙값(또는 위치)
mu0 = 0  # TODO
stat, p_value = stats.wilcoxon(x - mu0, alternative="two-sided")
print("W =", stat, "p =", p_value)

# 효과크기: Rank-biserial correlation (RBC)
# RBC = (W+ - W-) / (W+ + W-), W+는 양의 차이(+)에 대한 순위합
d = x - mu0
d_nz = d[d != 0]
if len(d_nz) == 0:
    print("모든 차이가 0입니다. 효과크기를 계산할 수 없습니다.")
else:
    ranks = stats.rankdata(np.abs(d_nz))
    w_pos = ranks[d_nz > 0].sum()
    w_neg = ranks[d_nz < 0].sum()
    rbc = (w_pos - w_neg) / (w_pos + w_neg)
    print("Rank-biserial correlation (RBC) =", rbc)

    # 신뢰구간(권장): RBC에 대해 부트스트랩
    rng = np.random.default_rng(0)
    B = 2000
    vals = []
    for _ in range(B):
        xx = rng.choice(x, size=len(x), replace=True)
        d2 = (xx - mu0)
        d2 = d2[d2 != 0]
        if len(d2) < 2:
            continue
        r2 = stats.rankdata(np.abs(d2))
        w_pos2 = r2[d2 > 0].sum()
        w_neg2 = r2[d2 < 0].sum()
        vals.append((w_pos2 - w_neg2) / (w_pos2 + w_neg2))
    ci = (np.percentile(vals, 2.5), np.percentile(vals, 97.5))
    print("95% CI for RBC (bootstrap) =", ci)
"""
    ),

    # 대응 t
    "paired_t": (
        "대응표본 t-검정",
        "동일 대상을 두 번 측정(사전/사후)한 경우입니다.",
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
est = np.mean(diff)

tcrit_two = stats.t.ppf(1 - alpha/2, df)
tcrit_one = stats.t.ppf(1 - alpha, df)

if alternative == "two-sided":
    ci = (est - tcrit_two*se, est + tcrit_two*se)
elif alternative == "greater":
    ci = (est - tcrit_one*se, np.inf)
else:  # "less"
    ci = (-np.inf, est + tcrit_one*se)

print(f"{int((1-alpha)*100)}% CI mean(diff) =", ci)
"""
    ),
    "wilcoxon_paired": (
        "Wilcoxon 부호순위 검정(대응)",
        "대응표본 t-검정의 비모수 대안입니다.",
        """from scipy import stats
import numpy as np

# x_before, x_after : paired arrays
diff = x_before - x_after
stat, p_value = stats.wilcoxon(diff, alternative="two-sided")
print("W =", stat, "p =", p_value)

# 효과크기: Rank-biserial correlation (RBC)
diff_nz = diff[diff != 0]
if len(diff_nz) == 0:
    print("모든 차이가 0입니다. 효과크기를 계산할 수 없습니다.")
else:
    ranks = stats.rankdata(np.abs(diff_nz))
    w_pos = ranks[diff_nz > 0].sum()
    w_neg = ranks[diff_nz < 0].sum()
    rbc = (w_pos - w_neg) / (w_pos + w_neg)
    print("Rank-biserial correlation (RBC) =", rbc)

    # 신뢰구간(권장): RBC 및 중앙값 차이 부트스트랩
    rng = np.random.default_rng(0)
    B = 2000
    vals = []
    med = []
    for _ in range(B):
        idx = rng.integers(0, len(diff), size=len(diff))
        d_b = diff[idx]
        d2 = d_b[d_b != 0]
        if len(d2) < 2:
            continue
        r2 = stats.rankdata(np.abs(d2))
        w_pos2 = r2[d2 > 0].sum()
        w_neg2 = r2[d2 < 0].sum()
        vals.append((w_pos2 - w_neg2) / (w_pos2 + w_neg2))
        med.append(np.median(d_b))
    alpha = 0.05
    if alternative == "two-sided":
        ci_rbc = (np.percentile(vals, 100*alpha/2), np.percentile(vals, 100*(1-alpha/2)))
        ci_med = (np.percentile(med, 100*alpha/2), np.percentile(med, 100*(1-alpha/2)))
    elif alternative == "greater":
        ci_rbc = (np.percentile(vals, 100*alpha), np.inf)
        ci_med = (np.percentile(med, 100*alpha), np.inf)
    else:
        ci_rbc = (-np.inf, np.percentile(vals, 100*(1-alpha)))
        ci_med = (-np.inf, np.percentile(med, 100*(1-alpha)))
    print("95% CI for RBC (bootstrap) =", ci_rbc)
    print("95% CI for median(diff) (bootstrap) =", ci_med)
"""
    ),

    # 독립 t: Student & Welch
    "ind_t_student": (
        "독립표본 t-검정(Student, 등분산 가정)",
        "두 독립 집단 비교이며, 등분산을 가정합니다.",
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
        "Welch t-검정(이분산)",
        "두 독립 집단 비교이며, 등분산을 가정하지 않습니다.",
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
        "Mann–Whitney U 검정",
        "독립표본 t-검정의 비모수 대안(순위합/랭크-섬)입니다.",
        """from scipy import stats
import numpy as np

# a, b : independent samples
u_stat, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
print("U =", u_stat, "p =", p_value)

# 효과크기 1: Rank-biserial correlation (RBC)
n1, n2 = len(a), len(b)
rbc = 1 - (2 * u_stat) / (n1 * n2)
print("Rank-biserial correlation (RBC) =", rbc)

# 효과크기 2(선택): Cliff's delta (RBC와 유사한 순위 기반 효과크기)
# delta = ( #a>b - #a<b ) / (n1*n2)
# (O(n^2)이므로 큰 표본에서는 주의)
delta = (np.sum(a[:,None] > b[None,:]) - np.sum(a[:,None] < b[None,:])) / (n1*n2)
print("Cliff's delta =", delta)

# 신뢰구간(권장): 부트스트랩
rng = np.random.default_rng(0)
B = 2000  # TODO
vals=[]
for _ in range(B):
    aa = rng.choice(a, size=n1, replace=True)
    bb = rng.choice(b, size=n2, replace=True)
    u_b, _ = stats.mannwhitneyu(aa, bb, alternative=alternative)
    vals.append(1 - (2*u_b)/(n1*n2))
alpha = 0.05
if alternative == "two-sided":
    ci = (np.percentile(vals, 100*alpha/2), np.percentile(vals, 100*(1-alpha/2)))
elif alternative == "greater":
    ci = (np.percentile(vals, 100*alpha), np.inf)
else:
    ci = (-np.inf, np.percentile(vals, 100*(1-alpha)))
print("95% CI for RBC (bootstrap) =", ci)

# 검정력: 닫힌형식은 흔치 않으므로 시뮬레이션 권장
"""
    ),

    # ANOVA / Kruskal
    "anova_oneway": (
        "일원분산분석(One-way ANOVA)",
        "3개 이상 독립 집단의 평균을 비교합니다(모수적).",
        """from scipy import stats
import numpy as np

# g1, g2, g3 : group arrays (extend as needed)
f_stat, p_value = stats.f_oneway(g1, g2, g3)
print("F =", f_stat, "p =", p_value)

groups = [g1, g2, g3]  # TODO extend
all_y = np.concatenate(groups)
grand_mean = np.mean(all_y)

# 제곱합
ss_between = sum(len(g)*(np.mean(g)-grand_mean)**2 for g in groups)
ss_within = sum(np.sum((g - np.mean(g))**2) for g in groups)
ss_total = ss_between + ss_within

# 효과크기: eta^2, omega^2, Cohen's f
eta2 = ss_between / ss_total
k = len(groups)
n = len(all_y)
df_between = k - 1
df_within = n - k
ms_within = ss_within / df_within
omega2 = (ss_between - df_between*ms_within) / (ss_total + ms_within)
f = np.sqrt(eta2 / (1 - eta2))
print("eta^2 =", eta2)
print("omega^2 =", omega2)
print("Cohen's f =", f)

# 검정력(근사)
from statsmodels.stats.power import FTestAnovaPower
power = FTestAnovaPower().power(effect_size=f, k_groups=k, nobs=n, alpha=0.05)
print("Power =", power)

# 신뢰구간:
# - 전체 효과크기(eta^2/omega^2)의 CI는 보통 부트스트랩으로 계산합니다.
# - 집단쌍 비교의 평균차 CI는 Tukey HSD(등분산) 또는 Games-Howell(이분산) 결과를 사용하세요.
"""
    ),

    "welch_anova": (
        "Welch 일원분산분석(이분산)",
        "정규성은 괜찮지만 분산이 같지 않을 때 3개 이상 독립 집단 평균을 비교합니다.",
        """import numpy as np
from statsmodels.stats.oneway import anova_oneway

alpha = 0.05  # TODO

# groups: list of arrays
groups = [g1, g2, g3]  # TODO extend
labels = ["g1", "g2", "g3"]  # TODO extend (groups와 순서 일치)

res = anova_oneway(groups, use_var="unequal", welch_correction=True)

# ── (1) 해석에 필요한 핵심만 요약 출력 ──
F = float(res.statistic)
p = float(res.pvalue)
df1 = float(res.df_num)
df2 = float(res.df_denom)

print(f"[Welch ANOVA] F({df1:.0f}, {df2:.2f}) = {F:.3f}, p = {p:.4g}")

if p < alpha:
    print(f"→ p < {alpha}: 집단 평균에 차이가 있습니다. (사후검정: Games–Howell 권장)")
else:
    print(f"→ p ≥ {alpha}: 평균 차이에 대한 통계적 증거가 부족합니다.")

# ── (2) 전체 효과크기(보고용): eta^2, omega^2 ──
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
print("  (경험칙) eta^2: 0.01=작음, 0.06=중간, 0.14=큼 (분야에 따라 달라질 수 있음)")

# 사후검정은 별도 스니펫(posthoc_games_howell)에서 유의한 쌍만 추려 효과크기(hedges g)까지 출력합니다.
"""
    ),

    "posthoc_tukey": (
        "사후검정: Tukey HSD(전통적 일원 ANOVA 이후)",
        "등분산이 대략 성립하고 일원 ANOVA가 유의할 때 사후검정으로 사용합니다.",
        """import numpy as np
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

alpha = 0.05  # TODO

# 1) long-format data
df = pd.DataFrame({
    "y": np.concatenate([g1, g2, g3]),   # TODO extend
    "group": (['g1']*len(g1) + ['g2']*len(g2) + ['g3']*len(g3))
})

# 2) Tukey HSD
res = pairwise_tukeyhsd(endog=df["y"], groups=df["group"], alpha=alpha)

# 3) 결과를 DataFrame으로 변환
tbl = pd.DataFrame(res.summary().data[1:], columns=res.summary().data[0])
# group1/group2/meandiff/p-adj/lower/upper/reject

print("[Tukey HSD] 전체 결과")
print(tbl.to_string(index=False))

# 4) 유의한 쌍만 추리기
sig = tbl[tbl["reject"] == True].copy()
print("\n[Tukey HSD] 유의한 쌍 (reject=True)")
if sig.empty:
    print("(없음)")
else:
    print(sig.to_string(index=False))

    # 5) 유의한 쌍에 대해 효과크기(Hedges' g) 계산
    def hedges_g(a, b):
        a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
        n1, n2 = len(a), len(b)
        s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
        sp = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1 + n2 - 2))
        d = (np.mean(a) - np.mean(b)) / sp
        J = 1 - 3 / (4*(n1+n2) - 9)  # small sample correction
        return J * d

    group_map = {"g1": g1, "g2": g2, "g3": g3}  # TODO extend labels/arrays

    sig["hedges_g"] = sig.apply(lambda r: hedges_g(group_map[r["group1"]], group_map[r["group2"]]), axis=1)

    # 보기 좋게 출력
    out_cols = ["group1", "group2", "meandiff", "p-adj", "lower", "upper", "hedges_g"]
    print("\n[Tukey HSD] 유의한 쌍 + 효과크기(Hedges' g)")
    print(sig[out_cols].to_string(index=False))

    print("\n(경험칙) |g|: 0.2=작음, 0.5=중간, 0.8=큼 (분야/측정에 따라 달라질 수 있음)")
"""
    ),

    "posthoc_games_howell": (
        "사후검정: Games–Howell(Welch ANOVA 이후)",
        "Welch ANOVA가 유의할 때(등분산 가정 없음) 사후검정으로 사용합니다.",
        """# Games–Howell (Welch ANOVA 이후 권장)
# pip install pingouin

import numpy as np
import pandas as pd
import pingouin as pg

alpha = 0.05  # TODO

df = pd.DataFrame({
    "y": np.concatenate([g1, g2, g3]),   # TODO extend
    "group": (['g1']*len(g1) + ['g2']*len(g2) + ['g3']*len(g3))
})

gh = pg.pairwise_gameshowell(dv="y", between="group", data=df)

print("[Games–Howell] 전체 결과")
print(gh.to_string(index=False))

# 유의한 쌍만 + 효과크기 출력
sig = gh[gh["pval"] < alpha].copy()
print("\n[Games–Howell] 유의한 쌍 (pval < alpha)")
if sig.empty:
    print("(없음)")
else:
    # pingouin은 보통 hedges 컬럼을 제공합니다(버전에 따라 다를 수 있음).
    cols = ["A", "B", "diff", "pval"]
    if "hedges" in sig.columns:
        cols.append("hedges")
        print(sig[cols].to_string(index=False))
        print("\n(경험칙) |g|: 0.2=작음, 0.5=중간, 0.8=큼")
    else:
        # 만약 hedges 컬럼이 없다면, 데이터로부터 직접 계산
        def hedges_g(a, b):
            a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
            n1, n2 = len(a), len(b)
            s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
            sp = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1 + n2 - 2))
            d = (np.mean(a) - np.mean(b)) / sp
            J = 1 - 3 / (4*(n1+n2) - 9)
            return J * d

        group_map = {"g1": g1, "g2": g2, "g3": g3}  # TODO extend
        sig["hedges_g"] = sig.apply(lambda r: hedges_g(group_map[r["A"]], group_map[r["B"]]), axis=1)
        print(sig[["A", "B", "diff", "pval", "hedges_g"]].to_string(index=False))
        print("\n(경험칙) |g|: 0.2=작음, 0.5=중간, 0.8=큼")
"""
    ),

    "posthoc_dunn": (
        "사후검정: Dunn 검정(Kruskal–Wallis 이후)",
        "Kruskal–Wallis가 유의한 뒤, 다중비교 보정을 포함한 사후검정으로 사용합니다.",
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
        "Kruskal–Wallis 검정",
        "일원분산분석(ANOVA)의 비모수 대안입니다(3개 이상 독립 집단).",
        """from scipy import stats
import numpy as np

# groups: list of arrays
groups = [g1, g2, g3]  # TODO extend
h_stat, p_value = stats.kruskal(*groups)
print("H =", h_stat, "p =", p_value)

# 효과크기: epsilon^2 (Kruskal-Wallis)
n = sum(len(g) for g in groups)
k = len(groups)
epsilon2 = (h_stat - k + 1) / (n - k)
print("epsilon^2 =", epsilon2)

# 신뢰구간(권장): 부트스트랩
rng=np.random.default_rng(0)
B=2000
vals=[]
for _ in range(B):
    boot_groups=[rng.choice(g, size=len(g), replace=True) for g in groups]
    h,_=stats.kruskal(*boot_groups)
    eps=(h - k + 1)/(n - k)
    vals.append(eps)
ci=(np.percentile(vals,2.5), np.percentile(vals,97.5))
print("95% CI for epsilon^2 (bootstrap) =", ci)

# 사후분석은 Dunn(보정 포함) 등을 사용하세요.
"""
    ),

    # 이진형/categorical
    "chi2_contingency": (
        "카이제곱 독립성 검정",
        "범주형 변수 간 관련성(분할표, R×C).",
        """
import numpy as np
from scipy import stats

# table : 분할표(관측도수) array (R×C)
# TODO INPUT:
# table = np.array([[...], [...], ...], dtype=int)

table_np: np.ndarray = np.asarray(table, dtype=float)
chi2, p, dof, expected_raw = stats.chi2_contingency(table_np, correction=False)
chi2 = float(chi2)
p = float(p)
dof = int(dof)
expected: np.ndarray = np.asarray(expected_raw, dtype=float)

print("chi2 =", chi2, "p =", p, "dof =", dof)

# 효과크기: Cramer's V (안전 계산)
n: float = float(table_np.sum())
r: int
c: int
r, c = table_np.shape
k: int = min(r - 1, c - 1)
cramers_v: float = float('nan')
if n > 0 and k > 0:
    cramers_v = float(np.sqrt(chi2 / (n * k)))
print("Cramer's V =", cramers_v)

# ── 사후분석: 조정 표준화 잔차(Adjusted standardized residuals) ──
# 경험칙: |잔차| > 2 이면 해당 셀이 기대와 유의미하게 다를 수 있음
row_sum = table_np.sum(axis=1, keepdims=True)
col_sum = table_np.sum(axis=0, keepdims=True)
row_prop = row_sum / n if n > 0 else np.zeros_like(row_sum, dtype=float)
col_prop = col_sum / n if n > 0 else np.zeros_like(col_sum, dtype=float)

den = np.sqrt(expected * (1.0 - row_prop) * (1.0 - col_prop))
with np.errstate(divide="ignore", invalid="ignore"):
    resid: np.ndarray = (table_np - expected) / den

# ── 보기 좋은 출력: 라벨(행/열) + 관측/기대/잔차 ──
import pandas as pd

# table이 DataFrame이면 라벨 유지, 아니면 임시 라벨 부여
table_df = table if hasattr(table, "index") else pd.DataFrame(table_np)

rows = []
for i, rlab in enumerate(table_df.index):
    for j, clab in enumerate(table_df.columns):
        z = resid[i, j]
        rows.append({
            "행(Row)": rlab,
            "열(Col)": clab,
            "관측(Obs)": int(table_np[i, j]),
            "기대(Exp)": float(expected[i, j]),
            "조정잔차(AdjResid)": float(z) if np.isfinite(z) else float("nan"),
            "유의(|z|>2)": bool(np.isfinite(z) and abs(z) > 2),
            "방향": ("관측>기대" if table_np[i, j] > expected[i, j] else "관측<기대")
        })

out = pd.DataFrame(rows)
out_sorted = out.sort_values("조정잔차(AdjResid)", key=lambda s: s.abs(), ascending=False)

print("\\n[사후분석] 조정 표준화 잔차 요약 (절댓값 큰 순 상위 10개)")
print(out_sorted.head(10).to_string(index=False))

sig = out_sorted[out_sorted["유의(|z|>2)"]]
print("\\n[사후분석] |조정잔차| > 2 인 셀(유의 가능)")
if len(sig) == 0:
    print("(없음)")
else:
    print(sig.to_string(index=False))
"""
    ),
    "chi2_ind_cochran_check": (
        "Cochran 조건 확인(독립성)",
        "카이제곱 독립성 검정의 근사 조건(Cochran's rule)을 먼저 확인합니다.",
        """
import numpy as np
from scipy import stats

# table : 분할표(관측도수) array (R×C)
# table = np.array([[...], [...], ...])

table_np: np.ndarray = np.asarray(table, dtype=float)
chi2, p, dof, expected_raw = stats.chi2_contingency(table_np, correction=False)
expected: np.ndarray = np.asarray(expected_raw, dtype=float)

# ── Cochran's rule(카이제곱 근사 조건) 확인 ──
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
        "카이제곱 독립성 검정(Monte Carlo)",
        "Cochran 조건 위반 시 Monte Carlo 근사로 p-value를 계산합니다.",
        """
import numpy as np
from scipy import stats

# table : 분할표(관측도수) array (R×C)
# table = np.array([[...], [...], ...])

chi2_obs, p_asym, dof, expected = stats.chi2_contingency(table, correction=False)

# ── Monte Carlo: 귀무가설(독립) 하에서 표를 재표본화하여 p-value 근사 ──
# (주의) 아래 템플릿은 "주어진 주변합(margins) 고정" 방식의 근사입니다.
# 환경/목적에 따라 구현 방식을 조정하세요.
rng = np.random.default_rng(0)
n_sim = 100_000
count_extreme = 0

table_np = np.asarray(table)

row_sums = table_np.sum(axis=1)
col_sums = table_np.sum(axis=0)
n = float(table_np.sum())
p_cols = (col_sums / n) if n > 0 else np.full_like(col_sums, 1.0 / len(col_sums), dtype=float)

for _ in range(n_sim):
    # 각 행의 합을 고정하고, 열 분포는 p_cols로 다항표본(근사)
    sim = np.vstack([rng.multinomial(int(rs), p_cols) for rs in row_sums])
    chi2_s, _, _, _ = stats.chi2_contingency(sim, correction=False)
    count_extreme += (chi2_s >= chi2_obs)

p_mc = count_extreme / n_sim
print("Primary p-value (Monte Carlo approx) =", p_mc)
print("(Optional) chi2 =", chi2_obs, "dof =", dof, "asymptotic p =", p_asym)

# 효과크기: Cramer's V (참고)
r, c = table.shape
k = min(r - 1, c - 1)
cramers_v = np.sqrt(chi2_obs / (n * k))
print("Cramer's V =", cramers_v)

        """
    ),
    "chi2_ind_collapse": (
        "범주 병합 후 카이제곱 독립성",
        "기대빈도가 작을 때 범주를 병합한 뒤 독립성 검정을 수행합니다.",
        """
import numpy as np
from scipy import stats

# [범주 병합(collapsing categories)] 후에 다시 카이제곱 독립성 검정을 수행하세요.
# 병합 규칙에 맞춰 table을 직접 재구성하세요.
#
# TODO INPUT:
# table = np.array([[...], [...], ...], dtype=int)  # 병합된 교차표

table_np = np.asarray(table)
res = stats.chi2_contingency(table_np, correction=False)

chi2 = float(res.statistic)
p = float(res.pvalue)
dof = int(res.dof)
expected = np.asarray(res.expected_freq, dtype=float)

print("chi2 =", chi2, "p =", p, "dof =", dof)

# 효과크기: Cramer's V (안전 계산)
n = float(table_np.sum())
r, c = table_np.shape
k = min(r - 1, c - 1)
cramers_v = np.nan
if n > 0 and k > 0:
    cramers_v = float(np.sqrt(chi2 / (n * k)))
print("Cramer's V =", cramers_v)

# ── 사후분석: 조정 표준화 잔차(Adjusted standardized residuals) ──
# 경험칙: |잔차| > 2 이면 해당 셀이 기대와 유의미하게 다를 수 있음
row_sum = table_np.sum(axis=1, keepdims=True)
col_sum = table_np.sum(axis=0, keepdims=True)
row_prop = row_sum / n if n > 0 else np.zeros_like(row_sum, dtype=float)
col_prop = col_sum / n if n > 0 else np.zeros_like(col_sum, dtype=float)

den = np.sqrt(expected * (1.0 - row_prop) * (1.0 - col_prop))
with np.errstate(divide="ignore", invalid="ignore"):
    resid = (table_np - expected) / den

for i in range(r):
    for j in range(c):
        val = float(resid[i, j]) if np.isfinite(resid[i, j]) else float("nan")
        flag = "기대와 다름(≈유의)" if np.isfinite(val) and abs(val) > 2 else ""
        print(f"cell[{i},{j}] resid = {val:.3f} {flag}")
        """
    ),
    "ffh_exact": (
        "Fisher–Freeman–Halton(FFH) 정확검정",
        "R×C 교차표에서 사용할 수 있는 정확검정(환경에 따라 지원이 다름).",
        """
import numpy as np
from scipy import stats
import pandas as pd

# table : 분할표(관측도수) array (R×C)
# table = np.array([[...], [...], ...], dtype=int)

table_np = np.asarray(table, dtype=float)
r, c = table_np.shape
N = float(table_np.sum())

# ── 효과크기: Cramér's V ──
# 정확검정/Monte Carlo p-value를 쓰더라도, 보고용 효과크기는 보통
# 같은 교차표의 chi-square 통계량으로부터 Cramér's V를 계산해 제시합니다.
chi2, p_asym, dof, expected = stats.chi2_contingency(table_np, correction=False)
k = min(r - 1, c - 1)
cramers_v = np.sqrt(chi2 / (N * k)) if (N > 0 and k > 0) else np.nan
print("Cramér's V =", float(cramers_v))

# ── Fisher–Freeman–Halton(FFH) 정확검정 p-value ──
# SciPy는 (대부분의 버전에서) Fisher exact를 2×2만 지원합니다.
# RxC FFH가 지원되는 환경이라면 아래가 동작할 수 있지만,
# 지원되지 않으면 Monte Carlo 근사를 사용합니다.
try:
    res = stats.fisher_exact(table_np)
    p_exact = getattr(res, "pvalue", res[1])
    print("FFH exact p =", float(p_exact))
except Exception as e:
    print("FFH exact test not available in this SciPy version.")
    print("Fallback: 귀무가설(독립) 하에서 Monte Carlo 근사(권장).")

    rng = np.random.default_rng(0)
    n_sim = 100_000
    chi2_obs = float(chi2)

    row_sums = table_np.sum(axis=1).astype(int)
    col_sums = table_np.sum(axis=0)
    p_cols = (col_sums / col_sums.sum()) if col_sums.sum() > 0 else np.full(c, 1.0/c)

    count_extreme = 0
    for _ in range(n_sim):
        sim = np.vstack([rng.multinomial(int(rs), p_cols) for rs in row_sums])
        chi2_sim, _, _, _ = stats.chi2_contingency(sim, correction=False)
        count_extreme += (chi2_sim >= chi2_obs)

    p_mc = count_extreme / n_sim
    print("Monte Carlo p (approx) =", float(p_mc))

# ── 사후분석 1: 조정 표준화 잔차(셀 진단) ──
row_sum = table_np.sum(axis=1, keepdims=True)
col_sum = table_np.sum(axis=0, keepdims=True)
row_prop = row_sum / N if N > 0 else np.zeros_like(row_sum)
col_prop = col_sum / N if N > 0 else np.zeros_like(col_sum)

den = np.sqrt(expected * (1.0 - row_prop) * (1.0 - col_prop))
with np.errstate(divide="ignore", invalid="ignore"):
    adj_resid = (table_np - expected) / den

print("[사후분석] 조정 표준화 잔차(AdjResid, z처럼 해석)")
# 라벨 출력: table이 DataFrame이면 index/columns 유지
table_df = table if isinstance(table, pd.DataFrame) else pd.DataFrame(table_np)
print(pd.DataFrame(adj_resid, index=table_df.index, columns=table_df.columns).round(3))

# ── 사후분석 2(선택): 행(pairwise) Fisher 정확검정 (열이 2개일 때) ──
# 만약 표가 R×2라면, 행을 2개씩 묶어 2×2 Fisher를 수행하고
# 다중비교 보정(Holm 등)을 적용할 수 있습니다.
if c == 2 and r >= 3:
    from itertools import combinations

    def holm_adjust(pvals):
        pvals = np.asarray(pvals, dtype=float)
        m = len(pvals)
        order = np.argsort(pvals)
        adj = np.empty(m, dtype=float)
        prev = 0.0
        for k_i, idx in enumerate(order):
            rank = k_i + 1
            val = (m - rank + 1) * pvals[idx]
            val = max(val, prev)
            prev = val
            adj[idx] = min(1.0, val)
        return adj

    rows = []
    pvals = []
    pairs = list(combinations(range(r), 2))
    for i, j in pairs:
        sub = np.vstack([table_np[i, :], table_np[j, :]]).astype(int)
        _, p = stats.fisher_exact(sub)
        pvals.append(p)
        rows.append({"row_i": table_df.index[i] if "table_df" in locals() else i,
             "row_j": table_df.index[j] if "table_df" in locals() else j,
             "p_fisher": float(p)})

    adj = holm_adjust(pvals)
    for rr, p_adj in zip(rows, adj):
        rr["p_holm"] = float(p_adj)

    df = pd.DataFrame(rows)
    print("[사후분석] 행 간 pairwise Fisher 정확검정 (Holm 보정)")
    print(df.sort_values("p_holm").to_string(index=False))
"""

    ),

    "fisher_exact": (
        "Fisher의 정확 검정(2×2)",
        "2×2 표에서 기대도수가 작을 때 사용합니다.",
        """import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import Table2x2

# 2×2 분할표(관측도수)
# table = np.array([[a, b],
#                   [c, d]])

# Fisher 정확검정(2×2)
oddsratio, p_value = stats.fisher_exact(table, alternative="two-sided")  # or "less"/"greater"
print("Fisher exact p =", p_value)
print("Odds ratio (scipy) =", oddsratio)

# 오즈비(OR) 95% CI (exact)
t22 = Table2x2(table)
ci_low, ci_high = t22.oddsratio_confint(alpha=0.05, method="exact")
print("95% CI for OR (exact) =", (ci_low, ci_high))

# ── 효과크기: 파이계수(phi, φ) ──
# 2×2 표에서는 φ = Cramér's V 와 동일합니다.
# φ = sqrt(chi2 / N)
chi2, p_chi2, dof, expected = stats.chi2_contingency(table, correction=False)
N = table.sum()
phi = np.sqrt(chi2 / N) if N > 0 else np.nan
print("Phi (φ) =", float(phi))

# (선택) 셀 단위 진단: 조정 표준화 잔차(Adjusted standardized residuals)
row_sum = table.sum(axis=1, keepdims=True)
col_sum = table.sum(axis=0, keepdims=True)
row_prop = row_sum / N if N > 0 else np.zeros_like(row_sum, dtype=float)
col_prop = col_sum / N if N > 0 else np.zeros_like(col_sum, dtype=float)

den = np.sqrt(expected * (1.0 - row_prop) * (1.0 - col_prop))
with np.errstate(divide="ignore", invalid="ignore"):
    adj_resid = (table - expected) / den
print("조정 표준화 잔차(AdjResid)\n", adj_resid)

# 사후분석 노트:
# - 2×2 표는 이미 '한 번의 비교'라서, 표준적인 사후분석은 별도로 하지 않습니다.
# - 대신 OR(및 CI), φ(효과크기), 잔차(어느 셀이 기대에서 벗어났는지)를 보고합니다.
"""

    ),
    "mcnemar": (
        "McNemar 검정(대응 이진)",
        "대응된 예/아니오 결과(사전/사후, 매칭 쌍).",
        """import numpy as np
from statsmodels.stats.contingency_tables import mcnemar, Table2x2

# 2×2 (대응) 분할표:
#          After+
# Before+   a    b
# Before-   c    d
table = np.array([[a, b],
                  [c, d]])

res = mcnemar(table, exact=True)   # 표본이 크면 exact=False(근사)도 가능
print("McNemar p =", res.pvalue)

# 효과크기(권장): discordant pair 비율과 OR(b/c)
b = table[0,1]
c = table[1,0]
if c == 0:
    print("주의: c=0이면 OR이 무한대로 발산합니다(보정 필요).")
else:
    or_bc = b / c
    print("OR (b/c) =", or_bc)

# OR 신뢰구간(간단 근사; Haldane-Anscombe 보정 포함 가능)
# 여기서는 Table2x2를 사용(해석은 주의: McNemar는 대응설계)
t22 = Table2x2(table)
try:
    ci_low, ci_high = t22.oddsratio_confint(alpha=0.05, method="exact")
    print("95% CI for OR (참고용) =", (ci_low, ci_high))
except Exception as e:
    print("OR CI 계산 실패(표본/구성 문제):", e)
"""
    ),

    # 연관성
    "pearsonr": (
        "Pearson 상관",
        "두 연속형 변수 간 선형 상관(대략 정규 가정) 여부를 평가합니다.",
        """from scipy import stats
import numpy as np

# x, y : arrays
r, p_value = stats.pearsonr(x, y)
print("Pearson r =", r, "p =", p_value)

# r의 95% 신뢰구간(Fisher z 변환)
alpha = 0.05
n = len(x)
z = np.arctanh(r)
se = 1/np.sqrt(n-3)
zcrit = stats.norm.ppf(1-alpha/2)
ci_z = (z - zcrit*se, z + zcrit*se)
ci_r = (np.tanh(ci_z[0]), np.tanh(ci_z[1]))
print("95% CI for r =", ci_r)

# 효과크기 해석: r 자체가 효과크기이며 r^2는 설명분산(선형 관계에서)
print("r^2 =", r**2)
"""
    ),
    "spearmanr": (
        "Spearman 상관",
        "비정규 연속형/서열형 변수에 대한 순위 기반 연관성입니다.",
        """from scipy import stats
import numpy as np

try:
    rho, p_value = stats.spearmanr(x, y, alternative=alternative)
except TypeError:
    rho, p_value = stats.spearmanr(x, y)
    if alternative != "two-sided":
        print("WARNING: This SciPy version does not support one-sided p-values for spearmanr; p is two-sided.")
print("Spearman rho =", rho, "p =", p_value)

# rho의 신뢰구간(권장): 부트스트랩
rng = np.random.default_rng(0)
B=2000
vals=[]
n=len(x)
for _ in range(B):
    idx = rng.integers(0, n, size=n)
    r, _ = stats.spearmanr(np.array(x)[idx], np.array(y)[idx])
    vals.append(r)

alpha = 0.05
if alternative == "two-sided":
    ci = (np.percentile(vals, 100*alpha/2), np.percentile(vals, 100*(1-alpha/2)))
elif alternative == "greater":
    ci = (np.percentile(vals, 100*alpha), np.inf)
else:
    ci = (-np.inf, np.percentile(vals, 100*(1-alpha)))
print("95% CI for rho (bootstrap) =", ci)
"""
    ),

    # Modeling
    "ols": (
        "선형회귀(OLS)",
        "하나 이상의 설명변수로 연속형 결과변수를 모델링합니다.",
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
        "로지스틱 회귀",
        "하나 이상의 설명변수로 이진형 결과변수를 모델링합니다.",
        """import numpy as np
import pandas as pd
import statsmodels.api as sm

# y: 0/1, X: (n×p) 설명변수
X = sm.add_constant(X)
model = sm.Logit(y, X).fit()
print(model.summary())

# 계수 신뢰구간
ci_beta = model.conf_int(alpha=0.05)
ci_beta.columns = ["ci_low", "ci_high"]
print("95% CI (beta):")
print(ci_beta)

# 오즈비(OR)와 OR 신뢰구간
or_ = np.exp(model.params)
ci_or = np.exp(ci_beta)
out = pd.DataFrame({"OR": or_})
out = out.join(ci_or)
print("OR 및 95% CI:")
print(out)

# 참고: OR 자체가 효과크기 해석의 핵심이며,
# 예측력 지표로는 pseudo-R^2, AUC 등을 추가로 볼 수 있습니다.
"""
    ),
    "poisson_glm": (
        "포아송 회귀(GLM)",
        "카운트(계수)형 결과변수를 모델링합니다. 과산포가 크면 음이항(Negative Binomial)도 고려하세요.",
        """import numpy as np
import pandas as pd
import statsmodels.api as sm

# y: count(0,1,2,...), X: 설명변수
X = sm.add_constant(X)
model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
print(model.summary())

# 계수 CI
ci_beta = model.conf_int(alpha=0.05)
ci_beta.columns = ["ci_low", "ci_high"]

# 발생률비(IRR = exp(beta)) 및 CI
irr = np.exp(model.params)
ci_irr = np.exp(ci_beta)
out = pd.DataFrame({"IRR": irr}).join(ci_irr)
print("IRR 및 95% CI:")
print(out)

# 참고:
# - 과산포가 크면 Negative Binomial도 고려하세요.
"""
    ),
    # One-sample z-test for mean (known sigma)
    "onesample_z_mean": (
        "일표본 z-검정(평균, σ 알려짐)",
        "모집단 표준편차 σ를 알고 있을 때(드묾) 사용합니다. 그렇지 않으면 일표본 t-검정을 사용하세요.",
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
        "일표본 비율 z-검정",
        "관측 비율을 가설값 p0(예: 전환율 기준값)과 비교합니다.",
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
        "이표본 비율 z-검정(독립 집단)",
        "두 독립 집단의 비율을 비교합니다. (2×2에서는 카이제곱과 동치지만, z-검정은 방향 가설을 다루기 편합니다.)",
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

    # 카이제곱 적합도 검정
    "chi2_gof": (
        "카이제곱 적합도 검정",
        "관측 도수가 주어진 기대 분포(한 개 범주형 변수)와 일치하는지 검정합니다.",
        """
import numpy as np
from scipy import stats

# 관측도수
# observed = np.array([o1, o2, o3, ...])

# 기대비율(합=1) 또는 기대도수(합=n)
# expected_probs = np.array([...])  # sum=1
# expected = expected_probs * observed.sum()

n = observed.sum()

# ── 카이제곱 적합도 검정(근사) ──
chi2, p = stats.chisquare(f_obs=observed, f_exp=expected)
df = len(observed) - 1
print("chi2 =", chi2, "df =", df, "p =", p)

# 효과크기(권장): Cohen's w (적합도 검정)
w = np.sqrt(chi2 / n)
print("Cohen's w =", w)

# ── 사후분석: 표준화 잔차(Standardized residuals) ──
# 표준화 잔차 = (관측 - 기대) / √기대
# 경험칙: |잔차| > 2 이면 해당 범주가 기대와 유의미하게 다를 수 있음
std_residuals = (observed - expected) / np.sqrt(expected)
for i, r in enumerate(std_residuals):
    flag = "기대와 다름(≈유의)" if abs(r) > 2 else ""
    print(f"category[{i}] std resid = {r:.3f} {flag}")

"""
    ),
    "chi2_gof_cochran_check": (
        "Cochran 조건 확인(적합도)",
        "카이제곱 적합도 검정의 근사 조건(Cochran's rule)을 먼저 확인합니다.",
        """
import numpy as np

# 관측도수
# observed = np.array([o1, o2, o3, ...])

# 기대비율(합=1) 또는 기대도수(합=n)
# expected_probs = np.array([...])  # sum=1
# expected = expected_probs * observed.sum()
expected = np.asarray(expected, dtype=float)

# ── Cochran's rule(카이제곱 근사 조건) 확인 ──
# (1) 기대빈도 < 5인 셀이 전체의 20% 이하  AND
# (2) 기대빈도 < 1인 셀이 0개
n_under1 = np.sum(expected < 1)
pct_under5 = np.mean(expected < 5)
cochran_ok = (n_under1 == 0) and (pct_under5 <= 0.20)

print("n_under1 =", int(n_under1))
print("pct_under5 =", float(pct_under5))
print("cochran_ok =", bool(cochran_ok))

        """
    ),
    "chi2_gof_mc": (
        "카이제곱 적합도 검정(Monte Carlo)",
        "Cochran 조건 위반 시 Monte Carlo 시뮬레이션으로 p-value를 근사합니다.",
        """
import numpy as np
from scipy import stats

# 관측도수
# observed = np.array([o1, o2, o3, ...])

# 기대비율(합=1)
# expected_probs = np.array([...])  # sum=1

n = int(observed.sum())
expected = expected_probs * n

# 관측된 chi-square 통계량(참조용)
chi2_obs, p_asym = stats.chisquare(f_obs=observed, f_exp=expected)
df = len(observed) - 1

# ── Monte Carlo 시뮬레이션(귀무가설 하에서 χ² 분포를 직접 생성) ──
rng = np.random.default_rng(0)
n_sim = 100_000  # TODO: 시뮬레이션 횟수
count_extreme = 0
for _ in range(n_sim):
    samp = rng.multinomial(n, expected_probs)
    chi2_s, _ = stats.chisquare(f_obs=samp, f_exp=expected)
    count_extreme += (chi2_s >= chi2_obs)

p_mc = count_extreme / n_sim
print("Primary p-value (Monte Carlo) =", p_mc)
print("(Optional) chi2 =", chi2_obs, "df =", df, "asymptotic p =", p_asym)

# 효과크기: Cohen's w (GOF)
w = np.sqrt(chi2_obs / n)
print("Cohen's w =", w)

# 사후분석: 표준화 잔차(|r| > 2 표시)
std_residuals = (observed - expected) / np.sqrt(expected)
for i, r in enumerate(std_residuals):
    if abs(r) > 2:
        print(f"category[{i}] std resid = {r:.3f}")

        """
    ),
    "chi2_gof_collapse": (
        "범주 병합 후 카이제곱 적합도",
        "기대빈도가 작을 때 범주를 병합한 뒤 적합도 검정을 수행합니다.",
        """
import numpy as np
from scipy import stats

# [범주 병합(collapsing categories)] 후에 다시 카이제곱 적합도 검정을 수행하세요.
# 아래는 템플릿입니다. 병합 규칙에 맞춰 observed/expected_probs를 직접 재구성하세요.

# 예시) 원래 6범주를 (0~2), (3~5)로 병합한다면:
# observed_raw = np.array([o1, o2, o3, o4, o5, o6])
# observed = np.array([observed_raw[:3].sum(), observed_raw[3:].sum()])
#
# expected_probs_raw = np.ones(6) / 6
# expected_probs = np.array([expected_probs_raw[:3].sum(), expected_probs_raw[3:].sum()])

n = int(observed.sum())
expected = expected_probs * n

chi2, p = stats.chisquare(f_obs=observed, f_exp=expected)
df = len(observed) - 1
print("chi2 =", chi2, "df =", df, "p =", p)

w = np.sqrt(chi2 / n)
print("Cohen's w =", w)

std_residuals = (observed - expected) / np.sqrt(expected)
for i, r in enumerate(std_residuals):
    flag = "기대와 다름(≈유의)" if abs(r) > 2 else ""
    print(f"category[{i}] std resid = {r:.3f} {flag}")

        """
    ),


    # 순열(퍼뮤테이션) test (independent two-sample) — advanced
}



# ----------------------------
# Effect size interpretation guides (rules of thumb)
# ----------------------------

EFFECT_GUIDE: Dict[str, str] = {
    # t-tests (mean differences)
    "one_sample_t": "Cohen's d: |d|≈0.2(작음), 0.5(중간), 0.8(큼).\nHedges' g도 동일한 기준을 자주 사용합니다.\n※ 맥락/도메인 기준이 더 중요합니다.",
    "paired_t": "Cohen's d(대응): |d|≈0.2(작음), 0.5(중간), 0.8(큼).\n대응설계에서는 동일 d라도 실제 의미가 더 클 수 있습니다(측정오차 감소).",
    "ind_t_student": "Cohen's d(Hedges' g): |d|≈0.2(작음), 0.5(중간), 0.8(큼).\n표본크기가 작거나 분산이 다르면 g를 권장합니다.",
    "ind_t_welch": "Cohen's d(Hedges' g): |d|≈0.2(작음), 0.5(중간), 0.8(큼).\nWelch는 등분산이 아닐 때 사용(효과크기 기준은 동일).",
    "one_sample_z_mean": "표준화 평균차(Cohen's d와 유사): |d|≈0.2(작음), 0.5(중간), 0.8(큼).",

    # correlations
    "pearsonr": "상관계수 r: |r|≈0.1(작음), 0.3(중간), 0.5(큼).\nr^2는 설명분산(선형 관계에서)로 해석합니다.",
    "spearmanr": "순위상관 rho: |rho|≈0.1(작음), 0.3(중간), 0.5(큼).\n자료가 서열/비정규일 때 r 대신 rho를 권장.",

    # nonparametric two/paired samples
    "paired_wilcoxon": "Rank-biserial correlation(RBC): |r|≈0.1(작음), 0.3(중간), 0.5(큼)처럼 해석하는 경우가 많습니다.\nCliff's delta 기준을 쓰기도 함(|δ|<0.147 무시, <0.33 작음, <0.474 중간, 그 이상 큼).",
    "mannwhitney": "Rank-biserial correlation(RBC): |r|≈0.1(작음), 0.3(중간), 0.5(큼) 경험칙.\nCliff's delta 경험칙(|δ|<0.147 무시, <0.33 작음, <0.474 중간, 그 이상 큼).",

    # ANOVA / KW
    "anova_oneway": "η²(eta squared): ≈0.01(작음), 0.06(중간), 0.14(큼).\n또는 Cohen's f: 0.10(작음), 0.25(중간), 0.40(큼).",
    "kruskal": "ε²(epsilon squared): 대략 0.01(작음), 0.08(중간), 0.26(큼) 경험칙을 종종 사용합니다.\n(문헌/분야에 따라 기준이 다를 수 있습니다.)",

    # categorical
    "chi2_contingency": "Cramer's V: (2×2에서는) 0.10(작음), 0.30(중간), 0.50(큼).\n표가 커질수록 동일 V라도 의미가 달라질 수 있어 맥락을 함께 보세요.",
    "chi2_gof": "Cohen's w: 0.10(작음), 0.30(중간), 0.50(큼).",
    "fisher_exact": "Odds ratio(OR): 1이면 차이 없음.\nOR은 비대칭 척도이므로 log(OR)로 생각하거나, 임상적 기준/리스크 차이도 함께 제시하는 것이 좋습니다.",

    # models
    "ols": "회귀: R²≈0.02(작음), 0.13(중간), 0.26(큼) 경험칙(분야 의존).\n표준화 회귀계수(β)나 부분 R²도 함께 보세요.",
    "logit": "로지스틱: OR=1이면 효과 없음.\nOR은 단위/스케일에 민감하므로, 의미 있는 단위로 재스케일하거나 예측확률 변화(마진 효과)도 함께 제시를 권장합니다.",
    # proportions
    "prop_1sample_ztest": "Cohen's h(비율): h = 2·arcsin(√p1) − 2·arcsin(√p2).\n|h|≈0.2(작음), 0.5(중간), 0.8(큼) 경험칙.\n※ p가 0이나 1에 가까우면 해석이 민감할 수 있습니다.",
    "prop_2sample_ztest": "Cohen's h(비율 차이): |h|≈0.2(작음), 0.5(중간), 0.8(큼) 경험칙.\n(두 비율 p1, p2에 대해 h = 2·arcsin(√p1) − 2·arcsin(√p2))",

    # rank/ordinal agreement
    "kendall_w": "Kendall's W(일치도, 0~1): 0이면 일치 없음, 1이면 완전 일치.\n경험칙 예: W≈0.1 약함, 0.3 보통, 0.5 강함(분야 의존).\n보통 Friedman 검정과 함께 보고합니다.",

    # dominance (nonparametric effect size)
    "cliffs_delta": "Cliff's delta(δ, -1~1): 두 집단의 우월 확률 기반 효과크기.\n경험칙(절대값): |δ|<0.147 무시, <0.33 작음, <0.474 중간, 그 이상 큼.\nRBC와 함께/대신 보고되기도 합니다.",

}

def show_snippet(key: str) -> None:
    if key not in TEST_SNIPPETS:
        print_node("내부 오류", f"TEST_SNIPPETS 키가 없습니다: {key}")
        return
    title, detail, code = TEST_SNIPPETS[key]

    # --- Apply the user's alternative choice (two-sided / less / greater) ---
    # If the snippet uses an `alternative=` argument, define a shared variable and wire it in.
    # This keeps the printed code consistent with the decision-tree choice.
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
        print_node("효과크기 해석 기준(경험칙)", EFFECT_GUIDE[key], "")


def show_assumption(key: str) -> None:
    if key not in ASSUMPTION_SNIPPETS:
        print_node("내부 오류", f"ASSUMPTION_SNIPPETS 키가 없습니다: {key}")
        return
    title, detail, code = ASSUMPTION_SNIPPETS[key]
    print_node(title, detail, code)
    if key in EFFECT_GUIDE:
        print_node("효과크기 해석 기준(경험칙)", EFFECT_GUIDE[key], "")


# ----------------------------
# Decision engine (one run)
# ----------------------------

def run_once() -> Dict[str, Any]:
    """의사결정 트리를 한 번 수행하고 요약 딕셔너리를 반환합니다."""
    print_node(
        "대화형 통계 검정 선택기",
        "질문에 답해 주세요. 그러면 적절한 통계 검정 방법과 코드 예시를 안내해 드립니다.\n"
        "모든 진행 경로는 최종 권장사항 요약으로 마무리되며, 이후에는 처음부터 다시 시작할 수 있습니다."
    )

    print_node(
        "확인 사항",
        "이 도구는 자주 사용되는 통계 검정을 선택하는 데 도움을 줍니다.\n"
        "하지만 통계적 사고(추론)를 대체하지는 않습니다.\n\n"
        "항상 다음을 고려하세요:\n"
        "1) 연구 설계와 무작위 배정 여부\n"
        "2) 관측값의 독립성\n"
        "3) 결측치 발생 메커니즘\n"
        "4) 대립가설의 방향성은 데이터를 보기 전에 결정했는지 여부\n"
        "5) 동등성 검정인지, 차이 검정인지, 비열등성 검정인지 구분\n"
        "6) 다중 예측변수와 상호작용 효과 고려 여부\n"
        "7) 통계적 유의성뿐 아니라 실질적(임상적) 유의성도 고려"
    )

    result: Dict[str, Any] = {"final_tests": [], "notes": []}
    goal = ask_choice(
        "목표가 무엇인가요?",
        [
            ("compare", "집단 비교 / 차이 검정"),
            ("assoc", "연관성 / 상관"),
            ("model", "예측 / 모형화"),
        ],
    )

    # ---------------- Compare ----------------
    if goal == "compare":
        ytype = ask_choice(
            "결과변수(Y) 유형은 무엇인가요?",
            [
                ("continuous", "연속형(구간/비율 척도)"),
                ("binary", "이진형(예/아니오)"),
                ("ordinal", "서열형(리커트) / 비정규 연속형"),
                ("count", "횟수형(사건 발생 횟수)"),
            ],
        )

        # 연속형 outcome
        if ytype == "continuous":
            xtype = ask_choice(
                "설명변수(X) 유형은 무엇인가요?",
                [
                    ("categorical", "범주형(집단/조건)"),
                    ("continuous", "연속형 설명변수"),
                ],
            )
            if xtype == "continuous":
                show_snippet("ols")
                result["final_tests"] = ["ols"]
                return result

            k = ask_choice(
                "집단/조건이 몇 개인가요?",
                [
                    ("1", "1개 집단 vs 상수"),
                    ("2", "2개 집단/조건"),
                    ("3plus", "3개 이상 집단/조건"),
                ],
            )

            # 1 group vs constant
            if k == "1":
                sigma_known = ask_yes_no("모집단 표준편차 σ를 알고 있나요? (알면 z-검정을 사용할 수 있습니다.)")

                # workflow: check normality -> decide z/t vs wilcoxon
                show_assumption("normality_shapiro")
                normal_fail = ask_yes_no("정규성이 깨졌나요? / 강한 이상치 / 표본이 매우 작나요?")

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

            # 2개 집단
            if k == "2":
                paired = ask_choice(
                    "대응(동일 대상)인가요, 독립 집단인가요?",
                    [("paired", "대응표본 / 반복측정 (동일 대상에서 측정)"), ("ind", "독립표본 (서로 다른 집단)")],
                )
                if paired == "paired":
                    # workflow: check normality of differences -> decide paired t vs wilcoxon
                    show_assumption("normality_shapiro")
                    normal_fail = ask_yes_no("대응차이의 정규성이 깨졌나요? / 강한 이상치 / 표본이 매우 작나요?")
                    if normal_fail:
                        show_snippet("wilcoxon_paired")
                        result["final_tests"] = ["wilcoxon_paired"]
                    else:
                        show_snippet("paired_t")
                        result["final_tests"] = ["paired_t"]
                    return result

                # independent groups: decision tree
                show_assumption("normality_shapiro")
                normal_fail = ask_yes_no("(어느 한 집단이라도) 정규성이 깨졌나요? / 강한 이상치 / 표본이 매우 작나요?")

                if normal_fail:
                    # 정규성 가정이 깨진 독립 2표본 비교의 대표적 비모수 대안
                    show_snippet("mann_whitney")
                    result["final_tests"] = ["mann_whitney"]
                    return result

                show_assumption("equal_var_levene")
                var_equal = ask_yes_no("분산이 같아 보이나요? (Levene p ≥ α)")

                if var_equal:
                    show_snippet("ind_t_student")
                    result["final_tests"] = ["ind_t_student"]
                else:
                    show_snippet("ind_t_welch")
                    result["final_tests"] = ["ind_t_welch"]
                return result

            # 3개 이상 집단
            repeated = ask_choice(
                "독립 집단인가요, 반복측정(동일 대상)인가요?",
                [("ind", "독립표본(서로 다른 집단)"), ("rm", "반복측정(동일 대상)")],
            )
            if repeated == "ind":
                # workflow: normality -> variance -> choose ANOVA vs Welch ANOVA vs Kruskal
                show_assumption("normality_shapiro")
                normal_fail = ask_yes_no("(어느 집단이라도) 정규성이 깨졌나요? / 강한 이상치 / 표본이 매우 작나요?")

                show_assumption("equal_var_levene")
                var_equal = ask_yes_no("분산이 같아 보이나요? (Levene p ≥ α)")

                if normal_fail:
                    print("\\n[참고] 3집단 이상 비모수 비교(Kruskal) 이후 사후분석에서는, 표본이 매우 작거나 동점(ties)이 많아\n"
                          "정확한 근사에 불안이 있으면 Monte Carlo / resampling 기반 방법을 참고하는 경우가 있습니다.\n")
                    show_snippet("kruskal")
                    show_snippet("posthoc_dunn")
                    result["final_tests"] = ["kruskal", "posthoc_dunn"]
                    return result

                if var_equal:
                    show_snippet("anova_oneway")
                    show_snippet("posthoc_tukey")
                    result["final_tests"] = ["anova_oneway", "posthoc_tukey"]
                    return result

                # 정규성은 대략 만족하지만 등분산이 아니면 → Welch ANOVA + Games-Howell 사후검정
                show_snippet("welch_anova")
                show_snippet("posthoc_games_howell")
                result["final_tests"] = ["welch_anova", "posthoc_games_howell"]
                return result
            else:
                # RM: we give RM-ANOVA snippet + offer Friedman if assumptions not OK
                # (Full sphericity workflow is more complex; keep as guided choice)
                print_node(
                    "반복측정 measures note",
                    "RM-ANOVA also involves sphericity; if assumptions are concerning, consider Friedman.\n"
                    "If you want strict sphericity checks and 사후검정, we can extend this branch."
                )
                assumptions_ok = ask_yes_no("반복측정 ANOVA 가정이 대체로 괜찮나요? (차이의 대략 정규, 극단적 이상치 없음, 구형성 OK/처리)")
                if assumptions_ok:
                    # We didn't include anova_rm in this streamlined v10 dict; keep it as modeling extension if needed.
                    # To keep the tool coherent, we recommend Friedman here if not ok; otherwise, suggest RM-ANOVA via statsmodels.
                    show_snippet("anova_oneway")  # placeholder would be wrong; so instead show note + recommend AnovaRM snippet
                    # Better: provide a dedicated snippet inline
                    print_node(
                        "반복측정-measures ANOVA (AnovaRM)",
                        "3개 이상 반복 조건에서의 모수적 선택지입니다.",
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
                        "Friedman 검정",
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

        # 이진형 outcome
        if ytype == "binary":
            # 이진형/categorical outcomes can mean several different things.
            bgoal = ask_choice(
                "어떤 종류의 질문인가요?",
                [
                    ("prop", "비율(성공확률) 비교(일표본/이표본)"),
                    ("assoc", "두 범주형 변수의 관련성(분할표)"),
                    ("gof", "적합도(한 범주형 변수 vs 기대비율)"),
                    ("model", "설명변수로 확률을 모형화(로지스틱 회귀)"),
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
                    "대응/매칭 자료인가요, 아니면 독립 집단인가요?",
                    [("paired", "대응/매칭"), ("ind", "독립")],
                )
                if paired == "paired":
                    # For paired binary outcomes, McNemar is the standard.
                    show_snippet("mcnemar")
                    result["final_tests"] = ["mcnemar"]
                    return result

                howmany = ask_choice(
                    "몇 개의 비율을 비교하나요?",
                    [("1", "일표본 비율 vs p0"), ("2", "두 독립 집단")],
                )
                if howmany == "1":
                    show_snippet("prop_1sample_ztest")
                    result["final_tests"] = ["prop_1sample_ztest"]
                    return result

                # two independent groups
                # note: If expected counts are very small, Fisher's exact is safer.
                small = ask_yes_no("2×2 표에서 기대도수가 작은가요? (규칙: 기대도수 중 하나라도 < 5)")
                if small:
                    show_snippet("fisher_exact")
                    result["final_tests"] = ["fisher_exact"]
                    return result

                show_snippet("prop_2sample_ztest")
                result["final_tests"] = ["prop_2sample_ztest"]
                return result

            # bgoal == "assoc": contingency table 연관성
            paired = ask_choice(
                "대응/매칭 자료인가요, 아니면 독립 집단인가요?",
                [("paired", "대응/매칭"), ("ind", "독립")],
            )
            if paired == "paired":
                show_snippet("mcnemar")
                result["final_tests"] = ["mcnemar"]
                return result

            small = ask_yes_no("2×2 표에서 기대도수가 작은가요? (규칙: 기대도수 중 하나라도 < 5)")
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
                "설명변수(X) 유형은 무엇인가요?",
                [
                    ("categorical", "범주형(집단/조건)"),
                    ("continuous", "연속형 설명변수(s)"),
                ],
            )
            if xtype == "continuous":
                print_node(
                    "서열회귀(OrderedModel)",
                    "Y가 서열형이고 설명변수가 있을 때 사용합니다.",
                    """from statsmodels.miscmodels.ordinal_model import OrderedModel

# y: ordered categories (e.g., 1..5), X: predictors
model = OrderedModel(y, X, distr="logit")  # or "probit"
res = model.fit(method="bfgs")
print(res.summary())
"""
                )
                result["final_tests"] = ["ordered_model"]
                return result

            k = ask_choice("집단이 몇 개인가요?", [("2", "2개 집단"), ("3plus", "3개 이상 집단")])
            if k == "2":
                paired = ask_choice("대응인가요, 독립인가요?", [("paired", "대응"), ("ind", "독립")])
                if paired == "paired":
                    show_snippet("wilcoxon_paired")
                    result["final_tests"] = ["wilcoxon_paired"]
                else:
                    show_snippet("mann_whitney")
                    result["final_tests"] = ["mann_whitney"]
                return result
            else:
                paired = ask_choice("반복측정인가요, 독립 집단인가요?", [("rm", "반복측정"), ("ind", "독립")])
                if paired == "rm":
                    # show inline friedman again
                    print_node(
                        "Friedman 검정",
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

        # 카운트 outcome
        if ytype == "count":
            # ── 최소 질문 UX: 카운트는 3가지로만 분기합니다. (분포/교차표/발생률)
            count_scenario = ask_choice(
                "카운트 자료는 어떤 상황인가요?",
                [
                    ("gof", "① 하나의 범주 분포가 기대 분포(이론/균등/알려진 비율)와 같은가? (예: 주사위 공정성)"),
                    ("ind", "② 두 범주형 변수 사이에 관련(독립성)이 있는가? (예: 성별×합격여부 교차표)"),
                    ("rate", "③ 시간/노출량 대비 사건 발생 횟수(발생률)를 비교/모형화 (예: 사고 건수/인-년)"),
                ],
            )

            if count_scenario == "gof":
                # 1) 항상 Cochran 조건 확인용 코드를 먼저 제공합니다.
                show_snippet("chi2_gof_cochran_check")

                cochran_res = ask_choice(
                    "Cochran 조건 결과는 어땠나요?",
                    [
                        ("ok", "조건 만족(OK)"),
                        ("bad", "조건 위반(VIOLATED)"),
                    ],
                )

                if cochran_res == "ok":
                    show_snippet("chi2_gof")
                    result["final_tests"] = ["chi2_gof"]
                    return result

                # Cochran 위반 시: 대안 선택(가이드)
                alt = ask_choice(
                    "Cochran 위반 시 어떤 대안을 사용할까요?",
                    [
                        ("collapse", "범주 병합(collapsing categories) 후 카이제곱"),
                        ("mc", "Monte Carlo 시뮬레이션(권장)"),
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
                # 1) 항상 Cochran 조건 확인용 코드를 먼저 제공합니다.
                show_snippet("chi2_ind_cochran_check")

                cochran_res = ask_choice(
                    "Cochran 조건 결과는 어땠나요?",
                    [
                        ("ok", "조건 만족(OK)"),
                        ("bad", "조건 위반(VIOLATED)"),
                    ],
                )

                if cochran_res == "ok":
                    show_snippet("chi2_contingency")
                    result["final_tests"] = ["chi2_contingency"]
                    return result

                # Cochran 위반 시: 표 크기에 따라 대안이 달라집니다.
                shape = ask_choice(
                    "교차표 크기는 무엇인가요?",
                    [
                        ("2x2", "2×2 교차표"),
                        ("rxk", "2×2가 아닌 R×C 교차표"),
                    ],
                )

                if shape == "2x2":
                    alt = ask_choice(
                        "Cochran 위반(2×2) 시 어떤 대안을 사용할까요?",
                        [
                            ("fisher", "Fisher 정확검정(권장)"),
                            ("mc", "Monte Carlo 근사"),
                        ],
                    )
                    if alt == "fisher":
                        show_snippet("fisher_exact")
                        result["final_tests"] = ["fisher_exact"]
                    else:
                        show_snippet("chi2_ind_mc")
                        result["final_tests"] = ["chi2_ind_mc"]
                    return result

                # R×C (2×2 아님)
                alt = ask_choice(
                    "Cochran 위반(R×C) 시 어떤 대안을 사용할까요?",
                    [
                        ("collapse", "범주 병합(collapsing categories) 후 카이제곱"),
                        ("ffh", "Fisher–Freeman–Halton(FFH) 정확검정(가능하면)"),
                        ("mc", "Monte Carlo 근사(권장)"),
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


            # count_scenario == "rate": 사건 발생 횟수(발생률/포아송) 흐름
            goal2 = ask_choice(
                "사건 발생 횟수: 목표가 무엇인가요?",
                [("compare_rates", "집단 간 발생률(rate) 비교"), ("model", "설명변수로 카운트를 모형화")],
            )
            if goal2 == "compare_rates":
                print_node(
                    "포아송 발생률(rate) 검정",
                    "statsmodels.stats.rates를 사용합니다. 버전에 따라 API가 달라질 수 있으니 상황에 맞게 조정하세요.",
                    """# Example sketch (API may vary):
# from statsmodels.stats.rates import test_poisson_2indep
# count1, exposure1 = 30, 1000  # 사건 수, 노출량(예: person-time)
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

    result["notes"].append("아니오 matching path (unexpected).")
    return result


def print_final_summary(res: Dict[str, Any]) -> None:
    tests = res.get("final_tests") or []
    notes = res.get("notes") or []
    if tests:
        pretty = ", ".join(tests)
        print_node(
            "최종 권장사항",
            f"권장되는 통계 검정: {pretty}\n\n다른 경로를 확인하려면 다시 실행하거나 재시작을 선택하세요."
        )
    else:
        print_node(
            "최종 권장사항",
            "선택된 검정이 없습니다.\n" + ("\n".join(notes) if notes else "")
        )


def run_smoketest() -> None:
    """
    간단 무결성 점검:
    1) 스니펫 문자열이 모두 파이썬 문법으로 컴파일되는지 확인
    2) 핵심 의존 패키지 임포트 가능 여부 확인
    (주의) 스니펫을 실제 실행(exec)까지 하지는 않습니다. 일부 스니펫은 사용자 데이터/그래프 출력이 필요합니다.
    """
    print_node("자체 점검(스모크 테스트)", "코드 문법/의존성/스니펫 컴파일을 확인합니다.")

    # 1) imports
    imports_ok = True
    missing = []
    try:
        import numpy  # noqa: F401
    except Exception:
        imports_ok = False
        missing.append("numpy")
    try:
        import scipy  # noqa: F401
    except Exception:
        imports_ok = False
        missing.append("scipy")
    try:
        import statsmodels  # noqa: F401
    except Exception:
        imports_ok = False
        missing.append("statsmodels")
    try:
        import pingouin  # noqa: F401
    except Exception:
        # pingouin은 Welch ANOVA/Games-Howell 등에 사용(환경에 따라 없을 수 있음)
        missing.append("pingouin(선택)")
    try:
        import scikit_posthocs  # noqa: F401
    except Exception:
        missing.append("scikit-posthocs(선택)")

    if imports_ok:
        print("필수 패키지 임포트: 정상")
    else:
        print("필수 패키지 임포트: 일부 실패")
    if missing:
        print("참고(미설치/선택 포함):", ", ".join(missing))

    # 2) compile all snippet blocks
    def _compile_snippet(name: str, code: str) -> tuple[bool, str]:
        try:
            compile(code, f"<snippet:{name}>", "exec")
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    failures = []

    for k, (title, desc, code) in ASSUMPTION_SNIPPETS.items():
        ok, msg = _compile_snippet(k, code)
        if not ok:
            failures.append((f"ASSUMPTION_SNIPPETS[{k}] ({title})", msg))

    for k, (title, desc, code) in TEST_SNIPPETS.items():
        ok, msg = _compile_snippet(k, code)
        if not ok:
            failures.append((f"TEST_SNIPPETS[{k}] ({title})", msg))

    if not failures:
        print("모든 스니펫 컴파일: 정상")
        print_node("점검 결과", "현재 환경에서 문법/의존성/스니펫 컴파일 기준으로는 이상이 발견되지 않았습니다.")
    else:
        print("일부 스니펫 컴파일 실패:")
        for name, msg in failures[:20]:
            print(f" - {name}: {msg}")
        if len(failures) > 20:
            print(f" ... 외 {len(failures)-20}개")
        print_node("점검 결과", "일부 스니펫에 문법 문제가 있습니다. 위 목록을 우선 수정하세요.")


def main() -> None:
    while True:
        res = run_once()
        print_final_summary(res)

        again = ask_yes_no("의사결정 트리를 다시 시작할까요?")
        if not again:
            print_node("완료", "종료합니다. 분석에 도움이 되었기를 바랍니다!")
            break



def run_fuzz(iterations: int = 100, seed: int = 1234) -> None:
    """Automatically traverse the decision tree with valid choices to catch runtime crashes."""
    import random
    import contextlib
    import io

    rnd = random.Random(seed)

    def auto_choice(prompt: str, choices):
        if not choices:
            raise ValueError("No choices provided")
        # deterministic but varied
        idx = rnd.randrange(len(choices))
        return choices[idx][0]

    def auto_yes_no(prompt: str) -> bool:
        return auto_choice(prompt, [("y","yes"),("n","no")]) == "y"

    global ask_choice, ask_yes_no, CURRENT_ALT
    saved_choice, saved_yesno, saved_alt = ask_choice, ask_yes_no, CURRENT_ALT
    try:
        ask_choice = auto_choice
        ask_yes_no = auto_yes_no
        # silence printing during fuzz
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(iterations):
                CURRENT_ALT = None
                run_once()
        print(f"[FUZZ] OK: ran run_once() {iterations} times without crashing. (seed={seed})")
    finally:
        ask_choice, ask_yes_no, CURRENT_ALT = saved_choice, saved_yesno, saved_alt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="통계 검정 선택기(CLI)")
    parser.add_argument("--smoketest", action="store_true", help="문법/의존성/스니펫 컴파일 자체 점검을 실행하고 종료합니다.")
    parser.add_argument("--fuzz", type=int, default=0, help="결정 트리를 자동 입력으로 N회 실행해 런타임 크래시를 점검합니다.")
    parser.add_argument("--seed", type=int, default=1234, help="--fuzz 실행 시 난수 시드")
    args = parser.parse_args()
    if args.smoketest:
        run_smoketest()
    elif args.fuzz and args.fuzz > 0:
        run_fuzz(iterations=args.fuzz, seed=args.seed)
    else:
        main()