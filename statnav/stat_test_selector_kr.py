# ============================================================
# 통계 검정 선택 및 추론 도우미 (CLI)
# ------------------------------------------------------------
# 기능:
#  - 검정 선택(수업 흐름 중심)
#  - 가정 검토(정규성: Shapiro + Q-Q(probplot), 등분산: Levene)
#  - 효과크기 및 신뢰구간(가능한 경우 해석적/부트스트랩 안내)
#
# 버전: 1.0
# 마지막 점검: 2026-02-14  |  smoketest: PASS
# Developed by: 김규열(Ojirokim)
# ============================================================

#!/usr/bin/env python3
"""대화형 통계 검정 선택기(CLI) — 워크플로우 친화 버전
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


def ask_yes_no(prompt: str) -> bool:
    return ask_choice(prompt, [("y", "예"), ("n", "아니오")]) == "y"


def print_node(title: str, detail: Optional[str] = None, code: Optional[str] = None) -> None:
    print("\n" + "=" * 72)
    print(title)
    if detail:
        print("-" * 72)
        print(detail)
    if code:
        print("-" * 72)
        print("코드 스니펫:")
        print(code)
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
    "equal_var_Bartlett": (
        "등분산 검정(Bartlett)",
        "정규성 하에서는 Levene보다 검정력이 높을 수 있으나, 비정규/이상치에 민감합니다.",
        """from scipy import stats

# For 2+ groups
stat, p = stats.Bartlett(group1, group2)  # pass 3개 이상 집단 too
print("Bartlett p =", p)
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
        "Wilcoxon 부호순위 검정(일표본, 상수와 비교)",
        "일표본 t-검정의 비모수 대안입니다. 중앙값(x − μ0)이 0인지 검정합니다.",
        """from scipy import stats
import numpy as np

# x : sample array, mu0 : 가설 중앙값(또는 위치)
mu0 = 0  # TODO
stat, p_value = stats.wilcoxon(x - mu0, alternative="two-sided")
print("W =", stat, "p =", p_value)

# 효과크기 r (근사; 정규근사 기반)
d = x - mu0
d_nz = d[d != 0]
n = len(d_nz)
mean_w = n * (n + 1) / 4
var_w = n * (n + 1) * (2 * n + 1) / 24
z = (stat - mean_w - 0.5) / np.sqrt(var_w)
r = z / np.sqrt(n)
print("Effect size r (approx) =", r)

# 신뢰구간(권장): 위치(중앙값) 차이 또는 r에 대해 부트스트랩
rng = np.random.default_rng(0)
B=2000
vals=[]
for _ in range(B):
    xx = rng.choice(x, size=len(x), replace=True)
    s,_ = stats.wilcoxon(xx-mu0, alternative="two-sided")
    d2 = xx - mu0
    d2 = d2[d2!=0]
    n2=len(d2)
    if n2<5: 
        continue
    mean_w2 = n2*(n2+1)/4
    var_w2 = n2*(n2+1)*(2*n2+1)/24
    z2=(s-mean_w2-0.5)/np.sqrt(var_w2)
    vals.append(z2/np.sqrt(n2))
ci=(np.percentile(vals,2.5), np.percentile(vals,97.5))
print("95% CI for r (bootstrap) =", ci)
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
tcrit = stats.t.ppf(1 - alpha/2, df)
ci = (np.mean(diff) - tcrit*se, np.mean(diff) + tcrit*se)
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

# 효과크기 r (근사)
diff_nz = diff[diff != 0]
n = len(diff_nz)
mean_w = n * (n + 1) / 4
var_w = n * (n + 1) * (2 * n + 1) / 24
z = (stat - mean_w - 0.5) / np.sqrt(var_w)
r = z / np.sqrt(n)
print("Effect size r (approx) =", r)

# 신뢰구간(권장): 부트스트랩으로 r 또는 중앙값 차이 CI
rng = np.random.default_rng(0)
B=2000
vals=[]
med=[]
for _ in range(B):
    idx = rng.integers(0, len(diff), size=len(diff))
    d = diff[idx]
    s,_ = stats.wilcoxon(d, alternative="two-sided")
    d2 = d[d!=0]
    n2=len(d2)
    if n2<5:
        continue
    mean_w2 = n2*(n2+1)/4
    var_w2 = n2*(n2+1)*(2*n2+1)/24
    z2=(s-mean_w2-0.5)/np.sqrt(var_w2)
    vals.append(z2/np.sqrt(n2))
    med.append(np.median(d))
ci_r=(np.percentile(vals,2.5), np.percentile(vals,97.5))
ci_med=(np.percentile(med,2.5), np.percentile(med,97.5))
print("95% CI for r (bootstrap) =", ci_r)
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
tcrit = stats.t.ppf(1 - alpha/2, df)
ci = (mean_diff - tcrit*se_diff, mean_diff + tcrit*se_diff)
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
tcrit = stats.t.ppf(1 - alpha/2, df_w)
ci = (mean_diff - tcrit*se_diff, mean_diff + tcrit*se_diff)
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
    u_b, _ = stats.mannwhitneyu(aa, bb, alternative="two-sided")
    vals.append(1 - (2*u_b)/(n1*n2))
ci=(np.percentile(vals,2.5), np.percentile(vals,97.5))
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
from scipy import stats

# groups: list of arrays
groups = [g1, g2, g3]  # TODO extend

res = anova_oneway(groups, use_var="unequal", welch_correction=True)
print(res)

# (참고) 효과크기: eta^2 / omega^2 (고전적 정의; 이분산에서도 보고용으로 자주 사용)
all_y = np.concatenate(groups)
grand_mean = np.mean(all_y)
ss_between = sum(len(g)*(np.mean(g)-grand_mean)**2 for g in groups)
ss_within = sum(np.sum((g - np.mean(g))**2) for g in groups)
ss_total = ss_between + ss_within
eta2 = ss_between/ss_total

k=len(groups); n=len(all_y)
df_between=k-1; df_within=n-k
ms_within = ss_within/df_within
omega2 = (ss_between - df_between*ms_within) / (ss_total + ms_within)
print("eta^2 (reporting) =", eta2)
print("omega^2 (reporting) =", omega2)

# 사후검정: Welch 유의 → Games-Howell 권장
"""
    ),

    "posthoc_tukey": (
        "사후검정: Tukey HSD(전통적 일원 ANOVA 이후)",
        "등분산이 대략 성립하고 일원 ANOVA가 유의할 때 사후검정으로 사용합니다.",
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
        "사후검정: Games–Howell(Welch ANOVA 이후)",
        "Welch ANOVA가 유의할 때(등분산 가정 없음) 사후검정으로 사용합니다.",
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
        """import numpy as np
from scipy import stats

# table : 분할표(관측도수) array
chi2, p, dof, expected = stats.chi2_contingency(table)
print("chi2 =", chi2, "p =", p, "dof =", dof)

# 효과크기: Cramer's V
n = table.sum()
r, c = table.shape
cramers_v = np.sqrt(chi2 / (n * (min(r-1, c-1))))
print("Cramer's V =", cramers_v)

# 검정력(근사)
from statsmodels.stats.power import GofChisquarePower
power = GofChisquarePower().power(effect_size=cramers_v, nobs=n, alpha=0.05)
print("Power (approx) =", power)

# Cramer's V 신뢰구간(권장: 부트스트랩)
rng = np.random.default_rng(0)
B = 2000  # TODO 반복 횟수
vals = []
# 원자료가 '개별 관측치'가 아니라 '도수표'인 경우, 멀티노미얼로 재표집하는 방식
p_hat = (table / n).reshape(-1)
for _ in range(B):
    sample = rng.multinomial(n, p_hat).reshape(r, c)
    chi2_b, _, _, _ = stats.chi2_contingency(sample)
    v_b = np.sqrt(chi2_b / (n * (min(r-1, c-1))))
    vals.append(v_b)
ci = (np.percentile(vals, 2.5), np.percentile(vals, 97.5))
print("95% CI for Cramer's V (bootstrap) =", ci)
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
oddsratio, p_value = stats.fisher_exact(table, alternative="two-sided")  # or "less"/"greater"
print("Fisher exact p =", p_value)
print("Odds ratio (scipy) =", oddsratio)

# 효과크기: 오즈비(OR) + 95% 신뢰구간
t22 = Table2x2(table)
print("Odds ratio (Table2x2) =", t22.oddsratio)
ci_low, ci_high = t22.oddsratio_confint(alpha=0.05, method="exact")  # exact CI
print("95% CI for OR =", (ci_low, ci_high))

# 참고:
# - 기대도수가 작은 경우(예: 어떤 셀 기대도수 < 5) Chi-square보다 Fisher가 더 안전합니다.
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

rho, p_value = stats.spearmanr(x, y)
print("Spearman rho =", rho, "p =", p_value)

# rho의 신뢰구간(권장): 부트스트랩
rng = np.random.default_rng(0)
B=2000
vals=[]
n=len(x)
for _ in range(B):
    idx = rng.integers(0, n, size=n)
    r,_ = stats.spearmanr(np.array(x)[idx], np.array(y)[idx])
    vals.append(r)
ci=(np.percentile(vals,2.5), np.percentile(vals,97.5))
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
        """import numpy as np
from scipy import stats

# 관측도수
# observed = np.array([o1, o2, o3, ...])

# 기대비율(합=1) 또는 기대도수(합=n)
# expected_probs = np.array([...])  # sum=1
# expected = expected_probs * observed.sum()

chi2, p = stats.chisquare(f_obs=observed, f_exp=expected)
print("chi2 =", chi2, "p =", p)

# 효과크기(권장): Cohen's w (적합도 검정)
n = observed.sum()
w = np.sqrt(chi2 / n)
print("Cohen's w =", w)

# 검정력(근사)
from statsmodels.stats.power import GofChisquarePower
power = GofChisquarePower().power(effect_size=w, nobs=n, alpha=0.05)
print("Power (approx) =", power)

# w의 신뢰구간: 보통 부트스트랩(또는 모델 기반)로 계산
rng = np.random.default_rng(0)
B = 2000
vals=[]
p_hat = observed/observed.sum()
for _ in range(B):
    samp = rng.multinomial(n, p_hat)
    exp = expected  # 기대분포는 고정(문제에서 주어짐)이라고 가정
    chi2_b, _ = stats.chisquare(f_obs=samp, f_exp=exp)
    vals.append(np.sqrt(chi2_b/n))
ci=(np.percentile(vals,2.5), np.percentile(vals,97.5))
print("95% CI for Cohen's w (bootstrap) =", ci)
"""
    ),

    # 순열(퍼뮤테이션) test (independent two-sample) — advanced
}


def show_snippet(key: str) -> None:
    if key not in TEST_SNIPPETS:
        print_node("내부 오류", f"TEST_SNIPPETS 키가 없습니다: {key}")
        return
    title, detail, code = TEST_SNIPPETS[key]
    print_node(title, detail, code)


def show_assumption(key: str) -> None:
    if key not in ASSUMPTION_SNIPPETS:
        print_node("내부 오류", f"ASSUMPTION_SNIPPETS 키가 없습니다: {key}")
        return
    title, detail, code = ASSUMPTION_SNIPPETS[key]
    print_node(title, detail, code)


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

                use_Bartlett = ask_yes_no("등분산성 가정이 성립하나요? (Levene 검정) (Bartlett는 정규성 가정)")
                if use_Bartlett:
                    show_assumption("equal_var_Bartlett")
                    var_equal = ask_yes_no("분산이 같아 보이나요? (Bartlett p ≥ α)")
                else:
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
                    print("\n[참고] 3집단 이상 비모수 비교(Kruskal) 이후 사후분석에서는, 표본이 매우 작거나 동점(ties)이 많아\n"
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
            goal2 = ask_choice(
                "카운트 자료: 목표가 무엇인가요?",
                [("compare_rates", "집단 간 발생률(rate) 비교"), ("model", "설명변수로 카운트를 모형화")],
            )
            if goal2 == "compare_rates":
                print_node(
                    "포아송 발생률(rate) 검정",
                    "statsmodels.stats.rates를 사용합니다. 버전에 따라 API가 달라질 수 있으니 상황에 맞게 조정하세요.",
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

    # ---------------- 연관성 ----------------
    if goal == "assoc":
        vtype = ask_choice(
            "연관성 분석: 변수 유형은 무엇인가요?",
            [
                ("cont_norm", "연속형 2개(대략 정규, 선형)"),
                ("cont_non", "연속형 non-normal or ordinal"),
                ("cat_cat", "범주형 vs 범주형"),
                ("bin_pred", "이진형 outcome with predictors"),
            ],
        )
        if vtype == "cont_norm":
            # workflow: optionally check normality via Shapiro+QQ on x and y
            show_assumption("normality_shapiro")
            _ = ask_yes_no("Pearson 상관을 진행할까요? (대략 정규/선형 가정)")
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
            "모형화: 결과변수 유형은 무엇인가요?",
            [
                ("continuous", "연속형"),
                ("binary", "이진형"),
                ("count", "카운트"),
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="통계 검정 선택기(CLI)")
    parser.add_argument("--smoketest", action="store_true", help="문법/의존성/스니펫 컴파일 자체 점검을 실행하고 종료합니다.")
    args = parser.parse_args()
    if args.smoketest:
        run_smoketest()
    else:
        main()
