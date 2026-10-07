# 🎬 Box Office Forecaster — Predicting Movie Revenue Before Release

**36103 Statistical Thinking for Data Science · Assignment 2 · Group 07 · UTS Spring 2026**

Can information known *before* a film is released predict how much it will earn at the box office, and flag likely US$100M blockbusters? This repo contains the full analysis pipeline (EDA → regression models → validation) and an interactive Streamlit app that turns the models into a decision tool.

▶️ **Demo video:** https://www.youtube.com/watch?v=DfvqqBvNwNA

---

## Research questions

| | Question | Model |
|---|---|---|
| **RQ1** | How much revenue variation can pre-release information explain, and which factors matter for standalone vs franchise films? | Linear regression on log revenue (+ Gamma GLM comparison) |
| **RQ2** | Can we predict before release whether a film will reach US$100M (only 19% of films do)? | Logistic regression with a cost-based threshold |

## Key results

| Metric | Value |
|---|---|
| Films analysed | 7,276 (98.4% of films with recorded revenue kept) |
| Revenue model (M4) test R² (log revenue) | **0.59** (train 0.56; 5-fold CV RMSE 1.71; LOOCV 1.70) |
| Budget elasticity | +10% budget → +5.6% revenue (originals), +3.9% (sequels) |
| Other drivers | Director track record +56%, earlier franchise success +22% / SD, Sep–Oct release −23% |
| Blockbuster classifier | AUC **0.91**, recall **0.83** at threshold 0.20 (vs 0.61 at 0.50) |
| Missed blockbusters (test set) | 108 → 48; expected cost −30% |

---

## Data

**The Movies Dataset** (Banik, 2017) — Kaggle: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

The raw data is **not included** in this repo (size and licence). Download it and place these three files in `data/raw/`:

```
data/raw/
├── movies_metadata.csv
├── credits.csv
└── keywords.csv
```

Post-release fields (popularity, vote count, vote average) are dropped to prevent leakage — the models only use information available before release.

---

## Quick start

Requires **Python 3.10+**.

```bash
git clone https://github.com/khainguyen04/movie_1.git
cd movie_1

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

python -m pip install -r requirements.txt
```

### 1. Run the analysis pipeline

```bash
python run_pipeline.py              # runs all 14 agents (EDA → models → export)
python run_pipeline.py --only 01 02 # run specific agents
python run_pipeline.py --from 10    # resume from agent 10 onwards
```

Outputs are written to `outputs/` (figures, tables, logs) and the trained models to `artifacts/`.

### 2. Launch the app

```bash
python -m streamlit run app/streamlit_app.py
```

Opens at http://localhost:8501. Stop with `Ctrl + C`.

### 3. Run the tests

```bash
pytest -q
```

---

## The interactive app

| Page | What it does |
|---|---|
| **Predict Revenue** | Enter a film's planned details → median revenue forecast, likely range, and a "why this forecast" breakdown of each factor's contribution |
| **Blockbuster Probability** | Chance of reaching US$100M+, shown against the 0.20 decision line |
| **Scenario Compare** | Side-by-side what-if analysis: change budget, release month or sequel status and see the forecast move |
| **About Model** | Plain-language explanation of the models, accuracy and limitations |

The app loads scikit-learn pipelines saved as `.joblib`, so it applies exactly the same cleaning, encoding and scaling as training.

---

## Pipeline: 14 agents

Each agent is a module in `agents/` exposing `run(state, cfg)`, orchestrated by `run_pipeline.py` and configured via `config.yaml`.

| # | Agent | Purpose |
|---|---|---|
| 01 | `agent_01_loader` | Load and merge metadata, credits and keywords |
| 02 | `agent_02_error_detection` | Malformed rows, duplicates, unit errors (values recorded in millions), unreliable revenue, leakage columns |
| 03 | `agent_03_missing_values` | Missing-value treatment: meaningful value ("Unknown"), median (runtime), model-based (budget regression + `budget_missing` flag) |
| 04 | `agent_04_descriptive` | Descriptive statistics — continuous (mean, SD, median, IQR, skewness) and categorical (counts, mode, proportions) |
| 05 | `agent_05_visualisation` | Distribution plots, scatter plots, boxplots |
| 06 | `agent_06_feature_engineering` | Sequel flag; time-aware director / cast / franchise track records (strictly earlier films only) |
| 07 | `agent_07_dependence` | Pearson correlation, mutual information, VIF, redundant-predictor checks |
| 08 | `agent_08_imbalance_split` | Class imbalance analysis, stratified 80/20 split, KS test of split balance |
| 09 | `agent_09_encode_scale` | One-hot encoding, standardisation on training statistics |
| 10 | `agent_10_regression` | OLS models M1–M4 (budget-only → full → interaction → backward elimination), Gamma GLM M5, AIC/BIC |
| 11 | `agent_11_classification` | Logistic regression, class weighting vs threshold tuning, 5:1 loss matrix |
| 12 | `agent_12_diagnostics_cv` | Assumption tests (Durbin–Watson, KS, Jarque–Bera, Rainbow, Breusch–Pagan, Cook's distance), HC3 robust SEs, 5-fold CV, LOOCV |
| 13 | `agent_13_reporter` | Summary tables, report figures and slide charts |
| 14 | `agent_14_export` | Save fitted pipelines and metadata to `artifacts/` for the app |

---

## Repository structure

```
movie_1/
├── config.yaml                 # paths, thresholds, model settings
├── run_pipeline.py             # orchestrates the 14 agents
├── requirements.txt
├── agents/                     # agent_01_loader.py … agent_14_export.py
├── utils/                      # io, parsing, plotting, metrics, shared feature logic, slide charts
├── app/
│   ├── streamlit_app.py        # app entry point
│   ├── pages/                  # 1_Predict_Revenue, 2_Blockbuster_Probability, 3_Scenario_Compare, 4_About_Model
│   ├── components/             # UI, charts, input widgets
│   ├── services/predictor.py   # loads models, scores inputs
│   └── assets/style.css
├── tests/                      # feature consistency + model loading tests
├── notebooks/
│   ├── build_notebook.py       # generates full_pipeline.ipynb
│   └── full_pipeline.ipynb
├── data/
│   ├── raw/                    # ← put the Kaggle CSVs here (not tracked)
│   ├── interim/
│   └── processed/
├── outputs/
│   ├── figures/                # 01_data_quality … 07_slides
│   ├── tables/
│   └── logs/
├── artifacts/                  # exported models, metadata, lookups
└── report/
    └── appendix_code.md        # code appendix generated by the notebook
```

---

## Methods at a glance

- **EDA:** data error detection, missing-value analysis, descriptive statistics, distribution / scatter / box plots, log transformation and standardisation, one-hot and target-style encoding, Pearson correlation and mutual information.
- **Revenue model:** linear regression on log revenue with a log budget × sequel interaction; backward elimination (p > 0.05) selected M4 (50 predictors), compared against a Gamma GLM with log link using AIC/BIC, a held-out test set, 5-fold CV and LOOCV.
- **Assumptions:** independence and linearity held; non-normal, heteroscedastic residuals addressed with the log transform, large sample size and HC3 robust standard errors.
- **Blockbuster model:** logistic regression; a missed blockbuster is treated as 5× costlier than a false alarm, and cross-validation selected a 0.20 threshold.

## Limitations

- Data end in 2017 and include only reported, nominal (not inflation-adjusted) revenues.
- 26.5% of budgets are imputed, which may weaken the estimated budget effect.
- Marketing spend, screen count and competing releases are not observed, which caps explained variation at about 59%.
- Franchise starters and first-time directors have no track record (cold start).
- Associations, not causal effects.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `zsh: command not found: streamlit` | Use `python -m streamlit run app/streamlit_app.py` inside the activated `.venv` |
| `ModuleNotFoundError: plotly` (or another package) | `python -m pip install -r requirements.txt` |
| App says model file not found | Run `python run_pipeline.py` (or `--only 14` if outputs already exist) |
| `Port 8501 is already in use` | `python -m streamlit run app/streamlit_app.py --server.port 8502` |

---

## Team — Group 07

Nguyen Minh Khai · Menghan · Gin-Yin Hsiao · Tianhao Yang · Jinyu Guo · Zihan Guo · Suyong Sun

## Reference

Banik, R. (2017). *The Movies Dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

---

*Academic project for 36103 Statistical Thinking for Data Science, University of Technology Sydney. Forecasts are for educational purposes and should support, not replace, expert judgement.*
