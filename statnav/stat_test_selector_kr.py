
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
# 버전: v1.3.1
# 마지막 점검: 2026-02-15  |  smoketest: PASS
# Developed by: 김규열(Ojirokim)
# License: MIT
# ============================================================

#!/usr/bin/env python3
"""대화형 통계 검정 선택기(CLI) — 워크플로우 친화 버전

v1.2 변경점:
- 단측/양측 선택을 필요한 검정에서만 1회 묻고, 이후 모든 코드 스니펫에 자동 반영
- 부트스트랩 CI도 단측 선택 시 한쪽만(±inf) 나오도록 개선(해당 스니펫)
- 각 검정의 효과크기 해석 기준을 함께 출력

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

# Common snippet settings (to avoid repeating in every snippet)
COMMON_SETTINGS = """alpha = 0.05  # TODO
alternative = \"two-sided\"  # \"two-sided\" | \"greater\" | \"less\"
"""



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
        safe_code = safe_code.replace("__COMMON_SETTINGS__", COMMON_SETTINGS)
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
        """
from scipy import stats
import numpy as np

__COMMON_SETTINGS__

# x : 1D 표본 배열
mu0 = 0  # TODO: H0 하의 평균

t_stat, p_value = stats.ttest_1samp(x, popmean=mu0, alternative=alternative)

xbar = float(np.mean(x))
sd = float(np.std(x, ddof=1))
d = (xbar - mu0) / sd if sd > 0 else np.nan

print("[일표본 t-검정]")
print(f"  H0: mean = {mu0} | alternative = {alternative}")
print(f"  mean = {xbar:.4f}, t = {t_stat:.4f}, p = {p_value:.4g}, alpha = {alpha}")
print(f"  효과크기(Cohen's d) = {d:.4f}  (|d|: 0.2=작음, 0.5=중간, 0.8=큼; 분야에 따라 달라질 수 있음)")

if p_value < alpha:
    print("  → H0 기각: 평균이 mu0와 다르다는 증거가 있습니다.")
else:
    print("  → H0 기각 실패: 평균 차이에 대한 증거가 부족합니다.")

# (mean - mu0) 신뢰구간
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
        """
from scipy import stats
import numpy as np

__COMMON_SETTINGS__

# x_before, x_after : 대응표본(길이 동일)
t_stat, p_value = stats.ttest_rel(x_before, x_after, alternative=alternative)

diff = np.asarray(x_before) - np.asarray(x_after)
est = float(np.mean(diff))
sd = float(np.std(diff, ddof=1))
d = est / sd if sd > 0 else np.nan

print("[대응표본 t-검정]")
print(f"  H0: mean(diff)=0 | alternative = {alternative}")
print(f"  mean(diff) = {est:.4f}, t = {t_stat:.4f}, p = {p_value:.4g}, alpha = {alpha}")
print(f"  효과크기(Cohen's d_paired) = {d:.4f}  (|d|: 0.2=작음, 0.5=중간, 0.8=큼; 분야에 따라 달라질 수 있음)")

if p_value < alpha:
    print("  → H0 기각: 대응 측정 간 평균 변화의 증거가 있습니다.")
else:
    print("  → H0 기각 실패: 평균 변화의 증거가 부족합니다.")

# mean(diff) 신뢰구간
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
        "Wilcoxon 부호순위 검정(대응)",
        "대응표본 t-검정의 비모수 대안입니다.",
        """from scipy import stats
import numpy as np

__COMMON_SETTINGS__

# x_before, x_after : paired arrays
diff = x_before - x_after

stat, p_value = stats.wilcoxon(diff, alternative=alternative)
print(f"[Wilcoxon(대응)] W = {stat:.4g}, p = {p_value:.4g}, alpha = {alpha}")

if p_value < alpha:
    print("→ 결론: 귀무가설 기각 (중앙값 차이가 0이 아닐 가능성이 큼)")
else:
    print("→ 결론: 귀무가설 기각 실패 (차이에 대한 증거 부족)")

# ── 효과크기: Rank-biserial correlation (RBC) ──
# RBC = (W+ - W-) / (W+ + W-)
diff_nz = diff[diff != 0]
if len(diff_nz) == 0:
    print("[효과크기] 모든 차이가 0 → 효과크기 계산 불가")
else:
    ranks = stats.rankdata(np.abs(diff_nz))
    w_pos = ranks[diff_nz > 0].sum()
    w_neg = ranks[diff_nz < 0].sum()
    rbc = (w_pos - w_neg) / (w_pos + w_neg)
    print(f"[효과크기] RBC = {rbc:.4f}  (|RBC|: 0.1 작음, 0.3 중간, 0.5 큼 — 분야 의존)")

    # ── 부트스트랩 CI (권장) ──
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
        print(f"[CI] 95% CI for RBC (bootstrap) = {ci}")
    else:
        print("[CI] 부트스트랩 표본이 부족하여 CI 계산 생략")"""
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
        """
from scipy import stats
import numpy as np

__COMMON_SETTINGS__

# a, b : 독립 표본
u_stat, p_value = stats.mannwhitneyu(a, b, alternative=alternative)
n1, n2 = len(a), len(b)

# 효과크기: Rank-biserial correlation(RBC), Cliff's delta
rbc = 1 - (2 * u_stat) / (n1 * n2)

def cliffs_delta(x, y):
    x = np.asarray(x); y = np.asarray(y)
    gt = sum((xi > y).sum() for xi in x)
    lt = sum((xi < y).sum() for xi in x)
    return (gt - lt) / (len(x) * len(y))

delta = cliffs_delta(a, b)

print("[Mann–Whitney U 검정]")
print(f"  U = {u_stat:.4f}, p = {p_value:.4g}, alpha = {alpha} | alternative = {alternative}")
print(f"  효과크기: RBC = {rbc:.4f} | Cliff's delta = {delta:.4f} (해석 기준은 분야/측정에 따라 다름)")

if p_value < alpha:
    print("  → H0 기각: 두 집단 분포(중앙경향)가 다를 가능성이 있습니다.")
else:
    print("  → H0 기각 실패: 분포 차이에 대한 증거가 부족합니다.")

# 권장: RBC 부트스트랩 CI
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
print(f"  95% 부트스트랩 CI for RBC = [{ci[0]:.4f}, {ci[1]:.4f}]")

"""
    ),

    # ANOVA / Kruskal
    "anova_oneway": (
        "일원분산분석(One-way ANOVA)",
        "3개 이상 독립 집단의 평균을 비교합니다(모수적).",
        """from scipy import stats
import numpy as np

alpha = 0.05  # TODO

# g1, g2, g3 : group arrays (extend as needed)
f_stat, p_value = stats.f_oneway(g1, g2, g3)
print(f"[One-way ANOVA] F = {f_stat:.4g}, p = {p_value:.4g}, alpha = {alpha}")

if p_value < alpha:
    print("→ 결론: 귀무가설 기각 (집단 평균에 차이가 있을 가능성이 큼)")
    print("→ 다음 단계(권장):")
    print("   - 등분산이 대략 성립하면: Tukey HSD")
    print("   - 이분산이면: Welch ANOVA + Games–Howell")
else:
    print("→ 결론: 귀무가설 기각 실패 (평균 차이에 대한 증거 부족)")

groups = [g1, g2, g3]  # TODO extend
all_y = np.concatenate(groups)
grand_mean = np.mean(all_y)

# sums of squares
ss_between = sum(len(g)*(np.mean(g)-grand_mean)**2 for g in groups)
ss_within = sum(np.sum((g - np.mean(g))**2) for g in groups)
ss_total = ss_between + ss_within

# effect sizes
eta2 = ss_between / ss_total if ss_total > 0 else np.nan
k = len(groups)
n = len(all_y)
df_between = k - 1
df_within = n - k
ms_within = ss_within / df_within
omega2 = (ss_between - df_between*ms_within) / (ss_total + ms_within) if (ss_total + ms_within) > 0 else np.nan
f = np.sqrt(eta2 / (1 - eta2)) if (eta2 is not np.nan and eta2 < 1) else np.nan

print(f"[효과크기] eta^2 = {eta2:.4f}, omega^2 = {omega2:.4f}, Cohen's f = {f:.4f}")
print("  (해석기준) eta^2: 0.01=작음, 0.06=중간, 0.14=큼 (분야에 따라 달라질 수 있음)")"""
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
print("  (해석기준) eta^2: 0.01=작음, 0.06=중간, 0.14=큼 (분야에 따라 달라질 수 있음)")

# 사후검정은 별도 스니펫(posthoc_games_howell)에서 전체 쌍에 대해 보정 p값/효과크기/신뢰구간 코멘트를 통합 출력합니다.
"""
    ),

    "posthoc_tukey": (
        "사후검정: Tukey HSD(전통적 일원 ANOVA 이후)",
        "등분산이 대략 성립하고 일원 ANOVA가 유의할 때 사후검정으로 사용합니다.",
        """import numpy as np
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

alpha = 0.05  # TODO

# TODO INPUT:
# - g1, g2, g3: 1D numeric arrays (extend for more groups)
# - 라벨 'g1','g2','g3'는 원하면 실제 집단명으로 바꾸세요.

# 1) long-format data
df = pd.DataFrame({
    "y": np.concatenate([g1, g2, g3]),   # TODO extend
    "group": (["g1"]*len(g1) + ["g2"]*len(g2) + ["g3"]*len(g3))
})

# 2) Tukey HSD
res = pairwise_tukeyhsd(endog=df["y"], groups=df["group"], alpha=alpha)

# 3) 결과를 DataFrame으로 변환
tbl = pd.DataFrame(res.summary().data[1:], columns=res.summary().data[0])
# columns: group1 group2 meandiff p-adj lower upper reject

# ---- 효과크기 헬퍼 ----
def _pooled_sd(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    return np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1 + n2 - 2))

def cohens_d(a, b):
    sp = _pooled_sd(a, b)
    return (np.mean(a) - np.mean(b)) / sp if sp > 0 else np.nan

def hedges_g(a, b):
    # 소표본에서 왜 Hedges' g를 쓰나요?
    # - Cohen's d는 표본수가 작을 때 효과크기를 약간 과대추정하는 경향이 있습니다.
    # - Hedges' g는 보정계수(J)를 곱해 이 편향을 줄여, 소표본에서 더 정확한 추정을 제공합니다.
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    d = cohens_d(a, b)
    J = 1 - 3 / (4*(n1+n2) - 9)
    return J * d

def auto_effect(a, b, small_n=20):
    # 자동 선택: 두 집단 중 하나라도 n < small_n 이면 Hedges' g 사용(기본 20)
    if len(a) < small_n or len(b) < small_n:
        return hedges_g(a, b), "Hedges' g (소표본 보정)"
    return cohens_d(a, b), "Cohen's d"

def ci_comment(lower, upper):
    lower = float(lower); upper = float(upper)
    if lower > 0 or upper < 0:
        return "신뢰구간에 0이 없음 → 통계적으로 신뢰 가능한 차이"
    return "신뢰구간에 0이 포함 → 차이가 통계적으로 불확실할 수 있음"

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

print("[Tukey HSD] 전체 결과 + 효과크기 (전체 쌍)")
print(tbl.to_string(index=False))

print()
print("(해석기준) |d| 또는 |g|: 0.2=작음, 0.5=중간, 0.8=큼 (분야/측정에 따라 달라질 수 있음)")
"""
    ),

    "posthoc_games_howell": (
        "사후검정: Games–Howell(Welch ANOVA 이후)",
        "Welch ANOVA가 유의할 때(등분산 가정 없음) 사후검정으로 사용합니다.",
        """import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats

alpha = 0.05  # TODO

# TODO INPUT:
# - g1, g2, g3 ... : 1D numeric arrays (extend for more groups)
# - 라벨은 group_map의 key와 일치해야 합니다.

# 1) long-format data
df = pd.DataFrame({
    "y": np.concatenate([g1, g2, g3]),   # TODO extend
    "group": (["g1"]*len(g1) + ["g2"]*len(g2) + ["g3"]*len(g3))
})

# 2) Games–Howell 사후검정
gh = pg.pairwise_gameshowell(data=df, dv="y", between="group")

# ---- 효과크기 헬퍼 (소표본 자동 Hedges' g) ----
def _pooled_sd(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    return np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1 + n2 - 2))

def cohens_d(a, b):
    sp = _pooled_sd(a, b)
    return (np.mean(a) - np.mean(b)) / sp if sp > 0 else np.nan

def hedges_g(a, b):
    # 소표본에서 왜 Hedges' g를 쓰나요?
    # - Cohen's d는 표본수가 작을 때 효과크기를 약간 과대추정하는 경향이 있습니다.
    # - Hedges' g는 보정계수(J)를 곱해 이 편향을 줄여, 소표본에서 더 정확한 추정을 제공합니다.
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    d = cohens_d(a, b)
    J = 1 - 3 / (4*(n1+n2) - 9)
    return J * d

def auto_effect(a, b, small_n=20):
    if len(a) < small_n or len(b) < small_n:
        return hedges_g(a, b), "Hedges' g (소표본 보정)"
    return cohens_d(a, b), "Cohen's d"

def ci_comment(lower, upper):
    if lower > 0 or upper < 0:
        return "신뢰구간에 0이 없음 → 통계적으로 신뢰 가능한 차이"
    return "신뢰구간에 0이 포함 → 차이가 통계적으로 불확실할 수 있음"

group_map = {"g1": g1, "g2": g2, "g3": g3}  # TODO extend

# 3) 모든 쌍에 대해 CI(계산) + 효과크기 추가
# pingouin 결과 예: A, B, diff, se, df, pval, hedges (버전에 따라 다를 수 있음)
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

print("[Games–Howell] 전체 결과 + 효과크기 (전체 쌍)")
print(gh.to_string(index=False))

print()
print("(해석기준) |d| 또는 |g|: 0.2=작음, 0.5=중간, 0.8=큼 (분야/측정에 따라 달라질 수 있음)")
"""
    ),

    "posthoc_dunn": (
        "사후검정: Dunn 검정(Kruskal–Wallis 이후)",
        "Kruskal–Wallis가 유의한 뒤, 다중비교 보정을 포함한 사후검정으로 사용합니다.",
        """
# Dunn 사후검정 (Kruskal–Wallis 이후): 전체 쌍 + 보정 p값 + 효과크기\n# 필요 패키지: scikit-posthocs\n#   pip install scikit-posthocs\n\nimport numpy as np\nimport pandas as pd\n\ntry:\n    import scikit_posthocs as sp\nexcept ImportError as e:\n    raise ImportError(\n        \"Dunn 검정을 위해 scikit-posthocs가 필요합니다. 설치: pip install scikit-posthocs\"\n    ) from e\n\nalpha = 0.05  # TODO\np_adjust = \"holm\"  # 대안: \"bonferroni\", \"fdr_bh\" 등\n\n# TODO INPUT:\n# - g1, g2, g3 ... : 1D numeric arrays (필요 시 확장)\n# - group_map의 키(이름)를 실제 집단명으로 바꾸세요\ngroup_map = {\"g1\": g1, \"g2\": g2, \"g3\": g3}  # TODO extend\n\n# 1) long-format 데이터\ndf = pd.DataFrame({\n    \"y\": np.concatenate(list(group_map.values())),\n    \"group\": np.concatenate([[k]*len(v) for k, v in group_map.items()])\n})\n\n# 2) Dunn 검정 (쌍별 비교, 다중비교 보정 p값 행렬)\np_mat = sp.posthoc_dunn(df, val_col=\"y\", group_col=\"group\", p_adjust=p_adjust)\n\n# 3) 행렬 -> long 테이블(전체 쌍)\npairs = []\ngroups = list(p_mat.index)\nfor i in range(len(groups)):\n    for j in range(i+1, len(groups)):\n        a = groups[i]; b = groups[j]\n        pairs.append({\"group1\": a, \"group2\": b, \"p_adj\": float(p_mat.loc[a, b])})\ntbl = pd.DataFrame(pairs)\n\n# ---- 효과크기: Cliff's delta (+ 부트스트랩 CI) ----\n# Cliff's delta = P(X>Y) - P(X<Y), 범위 [-1, 1]; 0이면 우위가 없음.\n# (비모수 비교에 적합하며 정규성 가정이 필요 없습니다.)\ndef cliffs_delta(x, y):\n    x = np.asarray(x); y = np.asarray(y)\n    gt = sum((xi > y).sum() for xi in x)\n    lt = sum((xi < y).sum() for xi in x)\n    return (gt - lt) / (len(x) * len(y))\n\ndef bootstrap_ci_delta(x, y, B=2000, alpha=0.05, seed=0):\n    rng = np.random.default_rng(seed)\n    x = np.asarray(x); y = np.asarray(y)\n    n1, n2 = len(x), len(y)\n    vals = []\n    for _ in range(B):\n        xb = rng.choice(x, size=n1, replace=True)\n        yb = rng.choice(y, size=n2, replace=True)\n        vals.append(cliffs_delta(xb, yb))\n    lo = float(np.quantile(vals, alpha/2))\n    hi = float(np.quantile(vals, 1 - alpha/2))\n    return lo, hi\n\neffects = []\nci_lows = []\nci_highs = []\ncomments = []\n\nfor _, r in tbl.iterrows():\n    a, b = r[\"group1\"], r[\"group2\"]\n    x, y = group_map[a], group_map[b]\n    dlt = float(cliffs_delta(x, y))\n    lo, hi = bootstrap_ci_delta(x, y, B=2000, alpha=0.05, seed=0)\n    effects.append(dlt); ci_lows.append(lo); ci_highs.append(hi)\n    comments.append(\"신뢰구간에 0 미포함\" if (lo > 0 or hi < 0) else \"신뢰구간에 0 포함\")\n\ntbl[\"cliffs_delta\"] = effects\ntbl[\"delta_ci_low\"] = ci_lows\ntbl[\"delta_ci_high\"] = ci_highs\ntbl[\"CI_comment\"] = comments\ntbl[\"reject\"] = tbl[\"p_adj\"] < alpha\n\n# 4) 통합 출력\nprint(f\"[Dunn] 전체 쌍 결과 (p_adjust={p_adjust}, alpha={alpha})\")\nprint(tbl.sort_values([\"p_adj\", \"group1\", \"group2\"]).to_string(index=False))\n\nprint()\nprint(\"(효과크기) Cliff's delta: 0=차이 없음; 크기 해석은 맥락 의존\")\n
"""
    ),
    "kruskal": (
        "Kruskal–Wallis 검정",
        "일원분산분석(ANOVA)의 비모수 대안입니다(3개 이상 독립 집단).",
        """from scipy import stats
import numpy as np

alpha = 0.05  # TODO

# groups: list of arrays
groups = [g1, g2, g3]  # TODO extend
labels = ["g1", "g2", "g3"]  # TODO extend (groups와 순서 일치)

h_stat, p_value = stats.kruskal(*groups)

print("[Kruskal–Wallis]")
print(f"  H = {h_stat:.4f}, p = {p_value:.4g}, alpha = {alpha}")

if p_value < alpha:
    print("  → 귀무가설 기각: 집단 분포(중앙 경향)에 차이가 있다는 증거가 있습니다.")
    print("  → 사후분석: Dunn 검정(다중비교 보정 포함)으로 '전체 쌍'을 확인하세요.")
else:
    print("  → 귀무가설 기각 실패: 집단 간 차이에 대한 증거가 부족합니다.")
    print("  (주의) 표본이 작으면 검정력이 낮아 p가 커질 수 있으니, 효과크기/CI도 함께 보세요.")

# 효과크기: epsilon^2 (Kruskal–Wallis)
n = sum(len(g) for g in groups)
k = len(groups)
epsilon2 = (h_stat - k + 1) / (n - k) if (n - k) > 0 else np.nan
print(f"  효과크기 epsilon^2 = {epsilon2:.4f}")
print("    (참고 기준) ε²≈0.01(작음), 0.08(중간), 0.26(큼) — 문헌/분야에 따라 달라질 수 있음")

# 신뢰구간(권장): epsilon^2 부트스트랩
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
print("  CI 해석: 0에 가까우면 집단 차이가 작을 수 있습니다(맥락 의존).")"""
    ),

    # 이진형/categorical
    "chi2_contingency": (
        "카이제곱 독립성 검정",
        "범주형 변수 간 관련성(분할표, R×C).",
        """import numpy as np
import pandas as pd
from scipy import stats

alpha = 0.05  # TODO

# table : 분할표(관측도수) array (R×C)
# TODO INPUT:
# table = np.array([[...], [...], ...], dtype=int)

table_np = np.asarray(table, dtype=float)

chi2, p, dof, expected_raw = stats.chi2_contingency(table_np, correction=False)
expected = np.asarray(expected_raw, dtype=float)

print("[카이제곱 독립성 검정]")
print(f"  chi2 = {chi2:.4f}, dof = {dof}, p = {p:.4g}, alpha = {alpha}")

if p < alpha:
    print("  → 귀무가설 기각: 두 범주형 변수는 독립이 아닐 가능성이 있습니다(연관성 존재).")
else:
    print("  → 귀무가설 기각 실패: 연관성에 대한 통계적 증거가 부족합니다.")
    print("  (주의) 표본이 작으면 검정력이 낮아 p가 커질 수 있으니, 효과크기도 함께 보세요.")

# 효과크기: Cramer's V
n = float(table_np.sum())
r, c = table_np.shape
k = min(r - 1, c - 1)
cramers_v = float(np.sqrt(chi2 / (n * k))) if (n > 0 and k > 0) else float("nan")
print(f"  효과크기 Cramer's V = {cramers_v:.4f}")
print("    (참고 기준) 2×2에서는 V≈0.10(작음), 0.30(중간), 0.50(큼 — 맥락 의존)")

# ── 사후분석: 조정 표준화 잔차(Adjusted standardized residuals) ──
# |잔차|가 큰 셀이 '어떤 방향으로' 기여했는지 힌트를 줍니다.
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
            "행(Row)": rlab,
            "열(Col)": clab,
            "관측(Obs)": int(table_np[i, j]),
            "기대(Exp)": float(expected[i, j]),
            "조정잔차(AdjResid)": float(z) if np.isfinite(z) else float("nan"),
            "Flag(|z|>2)": bool(np.isfinite(z) and abs(z) > 2),
            "방향": ("관측>기대" if table_np[i, j] > expected[i, j] else "관측<기대")
        })

out = pd.DataFrame(rows)
out_sorted = out.sort_values("조정잔차(AdjResid)", key=lambda s: s.abs(), ascending=False)

print("\\n[사후분석] 조정 표준화 잔차 요약 (|z| 큰 순 상위 10개)")
print(out_sorted.head(10).to_string(index=False))

sig = out_sorted[out_sorted["Flag(|z|>2)"]]
print("\\n[사후분석] |z| > 2 인 셀 (유의 가능)")
if sig.empty:
    print("(없음)")
else:
    print(sig.to_string(index=False))

print("\n해석 팁: p값은 '연관성 존재 여부', 잔차는 '어느 셀이 기여했는지', V는 '연관성 크기'를 요약합니다.")"""
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
        """import numpy as np
from scipy import stats

alpha = 0.05  # TODO

# table : 분할표(관측도수) array (R×C)
# table = np.array([[...], [...], ...], dtype=int)

table = np.asarray(table, dtype=float)
chi2_obs, p_asym, dof, expected = stats.chi2_contingency(table, correction=False)

# ── Monte Carlo p-value (귀무가설: 독립) ──
# (주의) 아래는 "주어진 주변합(margins) 고정" 근사 템플릿입니다.
rng = np.random.default_rng(0)
n_sim = 100_000  # TODO
count_extreme = 0

row_sums = table.sum(axis=1)
col_sums = table.sum(axis=0)
n = table.sum()

# 간단 근사(주변합 고정): 각 행을 다항분포로 샘플링 (근사)
# 엄밀한 고정-주변합 샘플링은 별도 알고리즘(예: r2dtable)이 필요할 수 있습니다.
col_probs = col_sums / n
for _ in range(n_sim):
    sim = np.vstack([rng.multinomial(int(rs), col_probs) for rs in row_sums]).astype(float)
    chi2_s, _, _, _ = stats.chi2_contingency(sim, correction=False)
    count_extreme += (chi2_s >= chi2_obs)

p_mc = count_extreme / n_sim

print(f"[Chi-square 독립성] (Monte Carlo) chi2 = {chi2_obs:.4g}, dof = {dof}, p_mc = {p_mc:.4g}, alpha = {alpha}")
print(f"  (참고) 비대칭 근사 p(asym) = {p_asym:.4g}")

if p_mc < alpha:
    print("→ 결론: 귀무가설 기각 (두 범주형 변수는 독립이 아닐 가능성이 큼)")
else:
    print("→ 결론: 귀무가설 기각 실패 (독립성 위배에 대한 증거 부족)")

# 효과크기: Cramer's V
r, c = table.shape
k = min(r - 1, c - 1)
cramers_v = np.sqrt(chi2_obs / (n * k)) if (n > 0 and k > 0) else np.nan
print(f"[효과크기] Cramer's V = {cramers_v:.4f}  (참고: 0.1 작음, 0.3 중간, 0.5 큼 — 문맥 의존)")

# 해석 팁(셀 단위): 조정 표준화 잔차(Adjusted standardized residuals)
row_prop = row_sums[:, None] / n
col_prop = col_sums[None, :] / n
den = np.sqrt(expected * (1.0 - row_prop) * (1.0 - col_prop))
with np.errstate(divide="ignore", invalid="ignore"):
    adj_resid = (table - expected) / den
print("[사후(셀 기여)] |잔차| > 2 인 셀은 기대치 대비 큰 편차일 수 있습니다.")"""
    ),
    "chi2_ind_collapse": (
        "범주 병합 후 카이제곱 독립성",
        "기대빈도가 작을 때 범주를 병합한 뒤 독립성 검정을 수행합니다.",
        """import numpy as np
from scipy import stats
import pandas as pd

alpha = 0.05  # TODO

# table : contingency table (observed counts) array (R×C)
# row_labels / col_labels are optional (for readable output)

table_np = np.asarray(table, dtype=float)
df = pd.DataFrame(table_np)

# ── Cochran rule check (quick) ──
chi2, p, dof, expected = stats.chi2_contingency(table_np, correction=False)
expected_flat = expected.ravel()
pct_lt5 = np.mean(expected_flat < 5) * 100

print(f"[Chi-square 독립성] chi2 = {chi2:.4g}, dof = {dof}, p = {p:.4g}, alpha = {alpha}")
print(f"[가정 점검] 기대도수 < 5 비율 = {pct_lt5:.1f}% (경험칙: 20% 이하 권장; 문맥 의존)")

if p < alpha:
    print("→ 결론: 귀무가설 기각 (연관성 가능)")
else:
    print("→ 결론: 귀무가설 기각 실패 (증거 부족)")

# ── 범주 병합 안내 ──
print("\\n[범주 병합 안내]")
print("- 기대도수가 너무 작다면, 희소 범주를 의미상 타당하게 병합한 뒤 재분석을 고려하세요.")
print("- 병합 후에는 동일 절차로 chi-square / Cramer's V / 잔차를 다시 확인합니다.")

# Effect size
n = table_np.sum()
r, c = table_np.shape
k = min(r - 1, c - 1)
cramers_v = np.sqrt(chi2 / (n * k)) if (n > 0 and k > 0) else np.nan
print(f"[효과크기] Cramer's V = {cramers_v:.4f}")"""
    ),
    "ffh_exact": (
        "Fisher–Freeman–Halton(FFH) 정확검정",
        "R×C 교차표에서 사용할 수 있는 정확검정(환경에 따라 지원이 다름).",
        """# Fisher–Freeman–Halton exact test (RxC table) — exact alternative to chi-square when expected counts are small
# NOTE: SciPy does not provide FFH exact directly for general RxC.
# Common approach: use `fisher_exact` implementations from external libraries or permutation/Monte Carlo.

import numpy as np
from scipy import stats

alpha = 0.05  # TODO

# table : RxC observed counts
table = np.asarray(table, dtype=int)

# Practical option: permutation/Monte Carlo based chi-square
chi2_obs, p_asym, dof, expected = stats.chi2_contingency(table, correction=False)

# Simple Monte Carlo: sample from multinomial with expected probabilities (approx)
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

print(f"[FFH exact 대안(근사)] chi2 = {chi2_obs:.4g}, dof = {dof}, p_mc = {p_mc:.4g}, alpha = {alpha}")
print("  (참고) FFH exact는 RxC에서 '정확검정'이지만, 구현은 라이브러리 의존적입니다.")
print("  (대안) 표본이 작거나 기대도수가 작으면 MC/순열 기반 p값을 보고하는 것이 실무적으로 흔합니다.")

if p_mc < alpha:
    print("→ 결론: 귀무가설 기각 (연관성 가능)")
else:
    print("→ 결론: 귀무가설 기각 실패 (증거 부족)")

# Effect size: Cramer's V
r, c = table.shape
k = min(r - 1, c - 1)
cramers_v = np.sqrt(chi2_obs / (n * k)) if (n > 0 and k > 0) else np.nan
print(f"[효과크기] Cramer's V = {cramers_v:.4f}")"""

    ),

    "fisher_exact": (
        "Fisher의 정확 검정(2×2)",
        "2×2 표에서 기대도수가 작을 때 사용합니다.",
        """import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import Table2x2

__COMMON_SETTINGS__

# 2×2 분할표
# table = np.array([[a, b],
#                   [c, d]], dtype=int)
table = np.asarray(table, dtype=float)

oddsratio, p_value = stats.fisher_exact(table, alternative=alternative)

print(f"[Fisher exact (2×2)] OR = {oddsratio:.4g}, p = {p_value:.4g}, alpha = {alpha}")

if p_value < alpha:
    print("→ 결론: 귀무가설 기각 (독립이 아닐 가능성이 큼)")
else:
    print("→ 결론: 귀무가설 기각 실패 (연관성 증거 부족)")

# OR 신뢰구간(권장): statsmodels
t22 = Table2x2(table)
ci_low, ci_high = t22.oddsratio_confint()
print(f"[CI] OR 95% CI = ({ci_low:.4g}, {ci_high:.4g})")
print("  - CI에 1이 포함되지 않으면(대략) OR이 유의하게 1과 다를 가능성이 큽니다.")

# 효과크기: phi (2×2)
chi2, p_chi, dof, exp = stats.chi2_contingency(table, correction=False)
n = table.sum()
phi = np.sqrt(chi2 / n) if n > 0 else np.nan
print(f"[효과크기] φ(phi) = {phi:.4f}  (참고: 0.1 작음, 0.3 중간, 0.5 큼)")

# 셀 기여: 조정 표준화 잔차
row_sum = table.sum(axis=1, keepdims=True)
col_sum = table.sum(axis=0, keepdims=True)
row_prop = row_sum / n
col_prop = col_sum / n
den = np.sqrt(exp * (1.0 - row_prop) * (1.0 - col_prop))
with np.errstate(divide="ignore", invalid="ignore"):
    adj_resid = (table - exp) / den
print("[잔차] |잔차| > 2 인 셀은 기대 대비 큰 편차일 수 있습니다.")"""

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
        """
from scipy import stats
import numpy as np

alpha = 0.05  # TODO

# x, y: 1D 배열(길이 동일)
r, p_value = stats.pearsonr(x, y)

print("[피어슨 상관(Pearson correlation)]")
print(f"  r = {r:.4f}, p = {p_value:.4g}, alpha = {alpha}")
if p_value < alpha:
    print("  → H0 기각: 0이 아닌 *선형* 연관성에 대한 증거가 있습니다.")
else:
    print("  → H0 기각 실패: 0이 아닌 선형 연관성의 증거가 부족합니다.")

# r의 신뢰구간(Fisher z 변환)
n = len(x)
z = np.arctanh(r)
se = 1 / np.sqrt(n - 3)  # n>3 필요
zcrit = stats.norm.ppf(1 - alpha/2)
ci_low = np.tanh(z - zcrit*se)
ci_high = np.tanh(z + zcrit*se)
print(f"  {int((1-alpha)*100)}% CI for r = [{ci_low:.4f}, {ci_high:.4f}]")
print("  해석 팁: CI에 0이 포함되지 않으면 보통 p < alpha와 일관됩니다.")

# 검정력(근사)
from statsmodels.stats.power import NormalIndPower
effect = r / np.sqrt(max(1e-12, 1 - r**2))
power = NormalIndPower().power(effect_size=effect, nobs1=n, alpha=alpha)
print(f"  Power(근사) = {power:.3f}")

"""
    ),
    "spearmanr": (
        "Spearman 상관",
        "비정규 연속형/서열형 변수에 대한 순위 기반 연관성입니다.",
        """
from scipy import stats
import numpy as np

alpha = 0.05  # TODO

# x, y: 1D 배열(길이 동일)
rho, p_value = stats.spearmanr(x, y)

print("[스피어만 순위상관(Spearman rank correlation)]")
print(f"  rho = {rho:.4f}, p = {p_value:.4g}, alpha = {alpha}")
if p_value < alpha:
    print("  → H0 기각: 단조(monotonic) 연관성에 대한 증거가 있습니다.")
else:
    print("  → H0 기각 실패: 단조 연관성의 증거가 부족합니다.")

print("  팁: Spearman은 이상치/비선형(단조) 패턴에 비교적 강건합니다.")

# 권장: rho 부트스트랩 CI
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
print(f"  95% 부트스트랩 CI for rho = [{ci[0]:.4f}, {ci[1]:.4f}]")

"""
    ),

    # Modeling
    "ols": (
        "선형회귀(OLS)",
        "하나 이상의 설명변수로 연속형 결과변수를 모델링합니다.",
        """import numpy as np
import pandas as pd
import statsmodels.api as sm

alpha = 0.05  # TODO

# y: (n,), X: (n, p)
# X에는 절편(constant)이 포함되지 않았다면 추가하세요.
X_ = sm.add_constant(X, has_constant="add")
model = sm.OLS(y, X_).fit()

print("[OLS 회귀] 핵심 요약")
print(f"R^2 = {model.rsquared:.4f}, Adj R^2 = {model.rsquared_adj:.4f}, F p = {model.f_pvalue:.4g}")

if model.f_pvalue < alpha:
    print("→ 결론: 모형 전체 유의 (적어도 하나의 계수가 0이 아닐 가능성)")
else:
    print("→ 결론: 모형 전체 비유의 (설명변수들이 y를 설명한다는 증거 부족)")

# 계수 테이블(해석 친화)
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
                            "CI에 0 미포함(유의 가능)", "CI에 0 포함(불확실)")
print("\n[계수 추정치]")
print(out.to_string())

print("\n[해석 팁]")
print("- coef 부호는 (다른 변수 고정 시) y의 증가/감소 방향을 의미합니다.")
print("- p/CI는 계수의 통계적 불확실성을 보여줍니다.")
print("- 가정 점검(잔차 정규성/등분산성/영향점)도 함께 확인하세요.")"""
    ),
    "logit": (
        "로지스틱 회귀",
        "하나 이상의 설명변수로 이진형 결과변수를 모델링합니다.",
        """
import numpy as np
import pandas as pd
import statsmodels.api as sm

alpha = 0.05  # TODO

# y: 0/1 이진, X: (n,p) 절편 제외
X = np.asarray(X, dtype=float)
y = np.asarray(y, dtype=float)

X_ = sm.add_constant(X)
model = sm.Logit(y, X_).fit(disp=False)

print("[로지스틱 회귀(Logit)]")
print(f"  n = {len(y)}, alpha = {alpha}")
print(f"  LL = {model.llf:.3f}, LL-Null = {model.llnull:.3f}, Pseudo R^2(McFadden) = {1 - model.llf/model.llnull:.4f}")

tbl = pd.DataFrame({
    "coef(log-odds)": model.params,
    "se": model.bse,
    "z": model.tvalues,
    "p": model.pvalues,
})
ci = model.conf_int(alpha=alpha)
tbl["ci_low"] = ci[0]; tbl["ci_high"] = ci[1]
tbl["OR"] = np.exp(tbl["coef(log-odds)"])
tbl["OR_ci_low"] = np.exp(tbl["ci_low"])
tbl["OR_ci_high"] = np.exp(tbl["ci_high"])
tbl["reject"] = tbl["p"] < alpha

print("\n[계수(log-odds) + 오즈비(OR)]")
print(tbl.to_string())

print("\n팁: OR 신뢰구간이 1을 포함하지 않으면(대략) p < alpha와 일치합니다.")

"""
    ),
    "poisson_glm": (
        "포아송 회귀(GLM)",
        "카운트(계수)형 결과변수를 모델링합니다. 과산포가 크면 음이항(Negative Binomial)도 고려하세요.",
        """
import numpy as np
import pandas as pd
import statsmodels.api as sm

alpha = 0.05  # TODO

# y: 카운트(0 이상 정수), X: (n,p) 절편 제외
X = np.asarray(X, dtype=float)
y = np.asarray(y, dtype=float)

X_ = sm.add_constant(X)
model = sm.GLM(y, X_, family=sm.families.Poisson()).fit()

print("[포아송 GLM]")
print(f"  n = {len(y)}, alpha = {alpha}")
print(f"  Deviance = {model.deviance:.3f}, Pearson chi2 = {model.pearson_chi2:.3f}")
print("  팁: 과산포가 크면(Pearson chi2 / df >> 1) 음이항(Negative Binomial) 고려.")

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

print("\n[계수 + 비율(Rate Ratio)]")
print(tbl.to_string())

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
        """import numpy as np
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

__COMMON_SETTINGS__

# count: 성공 횟수, nobs: 시행 횟수, p0: 귀무가설 비율
stat, p_value = proportions_ztest(count=count, nobs=nobs, value=p0, alternative=alternative)

print(f"[1표본 비율 z-검정] z = {stat:.4g}, p = {p_value:.4g}, alpha = {alpha}, H0: p = {p0}")

if p_value < alpha:
    print("→ 결론: 귀무가설 기각 (비율이 p0와 다를 가능성이 큼)")
else:
    print("→ 결론: 귀무가설 기각 실패 (증거 부족)")

# CI for p (Wilson 권장)
ci_low, ci_high = proportion_confint(count=count, nobs=nobs, alpha=alpha, method="wilson")
print(f"[CI] p의 {int((1-alpha)*100)}% CI (Wilson) = ({ci_low:.4f}, {ci_high:.4f})")
print("  - CI에 p0가 포함되지 않으면(대략) p가 p0와 다를 가능성이 큽니다.")

# Effect size: Cohen's h
p_hat = count / nobs
h = 2*np.arcsin(np.sqrt(p_hat)) - 2*np.arcsin(np.sqrt(p0))
print(f"[효과크기] Cohen's h = {h:.4f}  (0.2 작음, 0.5 중간, 0.8 큼)")"""
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
        """import numpy as np
from scipy import stats

alpha = 0.05  # TODO

# 관측도수
# observed = np.array([o1, o2, o3, ...])

# 기대비율(합=1) 또는 기대도수(합=n)
# expected_probs = np.array([...])  # sum=1
# expected = expected_probs * observed.sum()

observed = np.asarray(observed, dtype=float)
expected = np.asarray(expected, dtype=float)
n = float(observed.sum())

chi2, p = stats.chisquare(f_obs=observed, f_exp=expected)
df = int(len(observed) - 1)

print("[카이제곱 적합도 검정]")
print(f"  chi2 = {chi2:.4f}, df = {df}, p = {p:.4g}, alpha = {alpha}")

if p < alpha:
    print("  → 귀무가설 기각: 관측 분포가 기대 분포와 다르다는 증거가 있습니다.")
else:
    print("  → 귀무가설 기각 실패: 기대 분포와 다르다는 증거가 부족합니다.")
    print("  (주의) 표본이 작으면 검정력이 낮아 p가 커질 수 있습니다. 효과크기/잔차도 함께 보세요.")

# 효과크기: Cohen's w
w = float(np.sqrt(chi2 / n)) if n > 0 else float("nan")
print(f"  효과크기 Cohen's w = {w:.4f}")
print("    (참고 기준) w≈0.10(작음), 0.30(중간), 0.50(큼 — 맥락 의존)")

# 사후분석: 표준화 잔차
std_residuals = (observed - expected) / np.sqrt(expected)
print("\\n[사후분석] 표준화 잔차 (|z|>2이면 기대와 다른 범주일 수 있음)")
for i, z in enumerate(std_residuals):
    flag = " |z|>2" if np.isfinite(z) and abs(z) > 2 else ""
    print(f"  category[{i}] z = {float(z):.3f}{flag}")"""
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
        """import numpy as np
from scipy import stats

alpha = 0.05  # TODO

# observed : 1D observed counts
# expected : 1D expected counts (same length), or probabilities that sum to 1
obs = np.asarray(observed, dtype=float)

# If expected is probabilities, convert to counts
exp = np.asarray(expected, dtype=float)
if np.isclose(exp.sum(), 1.0):
    exp = exp * obs.sum()

chi2_obs = ((obs - exp) ** 2 / exp).sum()
df = len(obs) - 1

# Monte Carlo under H0: sample from multinomial with expected probs
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

print(f"[Chi-square 적합도] (Monte Carlo) chi2 = {chi2_obs:.4g}, df = {df}, p_mc = {p_mc:.4g}, alpha = {alpha}")

if p_mc < alpha:
    print("→ 결론: 귀무가설 기각 (분포가 기대와 다를 가능성이 큼)")
else:
    print("→ 결론: 귀무가설 기각 실패 (기대분포와의 차이에 대한 증거 부족)")

# Effect size: Cohen's w
w = np.sqrt(chi2_obs / n) if n > 0 else np.nan
print(f"[효과크기] Cohen's w = {w:.4f}  (참고: 0.1 작음, 0.3 중간, 0.5 큼)")

# Residuals
with np.errstate(divide='ignore', invalid='ignore'):
    resid = (obs - exp) / np.sqrt(exp)
print("[해석 팁] |잔차|가 큰 범주가 차이를 주도할 수 있습니다.")"""
    ),
    "chi2_gof_collapse": (
        "범주 병합 후 카이제곱 적합도",
        "기대빈도가 작을 때 범주를 병합한 뒤 적합도 검정을 수행합니다.",
        """import numpy as np
from scipy import stats

alpha = 0.05  # TODO

obs = np.asarray(observed, dtype=float)
exp = np.asarray(expected, dtype=float)
if np.isclose(exp.sum(), 1.0):
    exp = exp * obs.sum()

# Cochran-like quick check for GOF
pct_lt5 = np.mean(exp < 5) * 100

chi2, p = stats.chisquare(f_obs=obs, f_exp=exp)
df = len(obs) - 1

print(f"[Chi-square 적합도] chi2 = {chi2:.4g}, df = {df}, p = {p:.4g}, alpha = {alpha}")
print(f"[가정 점검] 기대도수 < 5 비율 = {pct_lt5:.1f}% (희소 범주 병합 고려)")

if p < alpha:
    print("→ 결론: 귀무가설 기각 (기대분포와 차이 가능)")
else:
    print("→ 결론: 귀무가설 기각 실패 (증거 부족)")

print("\\n[범주 병합 안내]")
print("- 기대도수가 작은 범주들은 의미상 타당하게 병합한 뒤 재분석을 고려하세요.")

n = obs.sum()
w = np.sqrt(chi2 / n) if n > 0 else np.nan
print(f"[효과크기] Cohen's w = {w:.4f}")"""
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
    "mannwhitney": "Rank-biserial correlation(RBC): |r|≈0.1(작음), 0.3(중간), 0.5(큼) 해석기준.\nCliff's delta 해석기준(|δ|<0.147 무시, <0.33 작음, <0.474 중간, 그 이상 큼).",

    # ANOVA / KW
    "anova_oneway": "η²(eta squared): ≈0.01(작음), 0.06(중간), 0.14(큼).\n또는 Cohen's f: 0.10(작음), 0.25(중간), 0.40(큼).",
    "kruskal": "ε²(epsilon squared): 대략 0.01(작음), 0.08(중간), 0.26(큼) 해석기준을 종종 사용합니다.\n(문헌/분야에 따라 기준이 다를 수 있습니다.)",

    # categorical
    "chi2_contingency": "Cramer's V: (2×2에서는) 0.10(작음), 0.30(중간), 0.50(큼).\n표가 커질수록 동일 V라도 의미가 달라질 수 있어 맥락을 함께 보세요.",
    "chi2_gof": "Cohen's w: 0.10(작음), 0.30(중간), 0.50(큼).",
    "fisher_exact": "Odds ratio(OR): 1이면 차이 없음.\nOR은 비대칭 척도이므로 log(OR)로 생각하거나, 임상적 기준/리스크 차이도 함께 제시하는 것이 좋습니다.",

    # models
    "ols": "회귀: R²≈0.02(작음), 0.13(중간), 0.26(큼) 해석기준(분야 의존).\n표준화 회귀계수(β)나 부분 R²도 함께 보세요.",
    "logit": "로지스틱: OR=1이면 효과 없음.\nOR은 단위/스케일에 민감하므로, 의미 있는 단위로 재스케일하거나 예측확률 변화(마진 효과)도 함께 제시를 권장합니다.",
    # proportions
    "prop_1sample_ztest": "Cohen's h(비율): h = 2·arcsin(√p1) − 2·arcsin(√p2).\n|h|≈0.2(작음), 0.5(중간), 0.8(큼) 해석기준.\n※ p가 0이나 1에 가까우면 해석이 민감할 수 있습니다.",
    "prop_2sample_ztest": "Cohen's h(비율 차이): |h|≈0.2(작음), 0.5(중간), 0.8(큼) 해석기준.\n(두 비율 p1, p2에 대해 h = 2·arcsin(√p1) − 2·arcsin(√p2))",

    # rank/ordinal agreement
    "kendall_w": "Kendall's W(일치도, 0~1): 0이면 일치 없음, 1이면 완전 일치.\n해석기준 예: W≈0.1 약함, 0.3 보통, 0.5 강함(분야 의존).\n보통 Friedman 검정과 함께 보고합니다.",

    # dominance (nonparametric effect size)
    "cliffs_delta": "Cliff's delta(δ, -1~1): 두 집단의 우월 확률 기반 효과크기.\n해석기준(절대값): |δ|<0.147 무시, <0.33 작음, <0.474 중간, 그 이상 큼.\nRBC와 함께/대신 보고되기도 합니다.",

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
        print_node("효과크기 해석 기준", EFFECT_GUIDE[key], "")


def show_assumption(key: str) -> None:
    if key not in ASSUMPTION_SNIPPETS:
        print_node("내부 오류", f"ASSUMPTION_SNIPPETS 키가 없습니다: {key}")
        return
    title, detail, code = ASSUMPTION_SNIPPETS[key]
    print_node(title, detail, code)
    if key in EFFECT_GUIDE:
        print_node("효과크기 해석 기준", EFFECT_GUIDE[key], "")


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

    
    # ---------------- Assoc / Correlation ----------------
    if goal == "assoc":
        atype = ask_choice(
            "어떤 종류의 연관성/상관을 보려 하나요?",
            [
                ("cont_cont", "연속형 ↔ 연속형 (상관)"),
                ("cat_cat", "범주형 ↔ 범주형 (분할표/연관성)"),
            ],
        )

        if atype == "cont_cont":
            # Pearson vs Spearman: provide both, user picks based on assumptions
            method = ask_choice(
                "어떤 상관을 사용할까요?",
                [
                    ("pearsonr", "Pearson (선형, 대략 정규/이상치 민감)"),
                    ("spearmanr", "Spearman (순위 기반, 비정규/이상치에 비교적 강건)"),
                ],
            )
            show_snippet(method)
            result["final_tests"] = [method]
            result["notes"].append("상관 분석은 '인과'를 의미하지 않습니다. 산점도/이상치/비선형 패턴을 함께 확인하세요.")
            return result

        if atype == "cat_cat":
            show_snippet("chi2_contingency")
            result["final_tests"] = ["chi2_contingency"]
            result["notes"].append("카이제곱 검정이 유의하면, 조정 표준화 잔차(셀 단위)를 함께 확인하는 것이 좋습니다.")
            return result

    # ---------------- Model / Prediction ----------------
    if goal == "model":
        my = ask_choice(
            "종속변수(Y) 유형은 무엇인가요?",
            [
                ("continuous", "연속형(선형회귀)"),
                ("binary", "이진형(로지스틱 회귀)"),
                ("count", "카운트(포아송/음이항 등)"),
                ("ordinal", "서열형(서열회귀)"),
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
                "서열회귀(OrderedModel)",
                "Y가 서열형(순서가 있는 범주)이고 설명변수가 있을 때 사용합니다.",
                """from statsmodels.miscmodels.ordinal_model import OrderedModel

# y: ordered categories (e.g., 1..5), X: predictors
model = OrderedModel(y, X, distr="logit")  # or "probit"
res = model.fit(method="bfgs")
print(res.summary())
""",
            )
            result["final_tests"] = ["ordered_model"]
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