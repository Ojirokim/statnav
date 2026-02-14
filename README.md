# StatNav

### Statistical Inference Navigator (CLI)

StatNav is a structured command-line statistical inference framework
designed to guide users through appropriate statistical test selection,
assumption checking, and effect size reporting.

This project emphasizes **statistical reasoning and workflow clarity**,
rather than black-box automation.

------------------------------------------------------------------------

<p align="center">
  <img src="assets/cat.gif" width="400">
</p>

## ✨ Features

-   Course-aligned statistical decision workflow
-   Parametric and nonparametric test selection
-   Assumption checks:
    -   Normality (Shapiro-Wilk + Q-Q plot via `scipy.stats.probplot`)
    -   Equal variance (Levene test)
-   Effect size computation
-   Confidence interval guidance (analytic or bootstrap where
    appropriate)
-   Binary, continuous, ordinal, and count outcome support
-   Logistic and Poisson regression pathways
-   Clear final recommendation summary in every branch

------------------------------------------------------------------------

## 🧠 Decision Workflow

The program follows a structured statistical logic:

Research Question\
→ Identify outcome type\
→ Identify predictor type\
→ Check assumptions\
→ Select appropriate test\
→ Report effect size + confidence interval

A full decision flowchart is available in:

docs/statistical_test_decision_tree.mmd

------------------------------------------------------------------------

## 🚀 Installation

Clone the repository:

``` bash
git clone https://github.com/Ojirokim/statnav.git
cd statnav
```

Create a virtual environment:

``` bash
python -m venv .venv
```

Activate it:

Windows

``` bash
.venv\Scripts\activate
```

Mac/Linux

``` bash
source .venv/bin/activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## ▶️ Running the Program

English Version:

``` bash
python -m statnav.stat_test_selector_en
```

Korean Version:

``` bash
python -m statnav.stat_test_selector_kr
```

------------------------------------------------------------------------

## 📁 Project Structure

statnav/

├── statnav/\
│ ├── **init**.py\
│ ├── stat_test_selector_en.py\
│ └── stat_test_selector_kr.py

├── docs/\
│ ├── statistical_test_decision_tree.mmd\
│ └── workflow.md

├── README.md\
├── LICENSE\
└── requirements.txt

------------------------------------------------------------------------

## 🎯 Design Philosophy

StatNav is designed as:

-   A learning-oriented inference framework
-   A personal statistical toolkit
-   A structured decision navigator

It is **not** intended to replace full statistical analysis pipelines or
domain expertise.

------------------------------------------------------------------------

## 📌 Roadmap (Future Improvements)

-   Modularization of statistical engines
-   Automated report generation
-   Optional GUI version
-   Web interface version
-   pip-installable package
-   Expanded bootstrap CI support

------------------------------------------------------------------------

## 📜 License

MIT License\
See LICENSE file for details.

------------------------------------------------------------------------

# 한국어 설명

## StatNav --- 통계 추론 네비게이터 (CLI)

StatNav는 통계 검정 선택과 추론 과정을 체계적으로 안내하기 위한 CLI 기반
통계 의사결정 도구입니다.

### 주요 기능

-   수업 흐름에 맞춘 통계 검정 선택
-   가정 검토:
    -   정규성 (Shapiro + Q-Q plot)
    -   등분산성 (Levene 검정)
-   효과크기 계산
-   신뢰구간 안내
-   연속형 / 이분형 / 순서형 / 계수형 결과 변수 지원
-   로지스틱 회귀 및 포아송 회귀 경로 포함

### 실행 방법

영어 버전:

``` bash
python -m statnav.stat_test_selector_en
```

한국어 버전:

``` bash
python -m statnav.stat_test_selector_kr
```

본 프로젝트는 자동화된 통계 엔진이 아니라 **학습 중심의 추론
프레임워크**로 설계되었습니다.
