# CMPE 255: Exploratory Data Analysis, Data Preparation & AutoML

Assignment for CMPE 255 (Data Mining), SJSU, Fall 2026.

| Part | What | Status |
|---|---|---|
| **Part 1** | Manual + automated EDA, data cleaning and preparation, clustering, anomaly removal, imputation, feature engineering and selection, and AutoML (AutoViML) on a **tabular** and a **time-series** Kaggle dataset | ✅ notebooks executed |
| **Part 2** | EDA done by an **AI data-analysis agent** instead of by hand | 🧩 scaffolded; needs an LLM API key to run |

## 🎥 Video walkthroughs

| Video | Link |
|---|---|
| Part 1 (i), tabular (Hotel Bookings): code walkthrough + user journey | _TODO: add YouTube link_ |
| Part 1 (ii), time series (Property Sales): code walkthrough + user journey | _TODO: add YouTube link_ |
| Part 2, agentic EDA | _TODO: add YouTube link_ |

---

## Part 1

| | (i) Tabular | (ii) Time series |
|---|---|---|
| **Dataset** | [Hotel booking demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) (119,390 × 32) | [House Property Sales Time Series](https://www.kaggle.com/datasets/htagholdings/property-sales) (29,580 sales, 2007–2019) |
| **Task** | Classify whether a booking is canceled | Forecast next week's median house price |
| **Notebook** | [`hotel_bookings_eda_prep_automl.ipynb`](part1/tabular_hotel_bookings/hotel_bookings_eda_prep_automl.ipynb) | [`property_sales_eda_prep_automl.ipynb`](part1/timeseries_property_sales/property_sales_eda_prep_automl.ipynb) |
| **Open in Colab** | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anita-agasaveeran/CMPE255_Exploratory_Data_Analysis/blob/main/part1/tabular_hotel_bookings/hotel_bookings_eda_prep_automl.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anita-agasaveeran/CMPE255_Exploratory_Data_Analysis/blob/main/part1/timeseries_property_sales/property_sales_eda_prep_automl.ipynb) |

### What each notebook covers

| Assignment step | Tabular: Hotel Bookings | Time series: Property Sales |
|---|---|---|
| **a) EDA and visualisation** | Schema and missingness audit, impossible values, **systematic leakage scan** (finds `required_car_parking_spaces` and the suspect `country`), **duplicate analysis** (duplicates are mostly cancelled group blocks), distributions, cancellation drivers, seasonality, countries, Spearman correlations | Coverage over time, price distributions, trend and volume, sales seasonality, postcode × year heatmap, STL decomposition, ADF stationarity tests, ACF/PACF, cross-check against the publisher's moving average |
| **Auto-EDA** | AutoViz | AutoViz |
| **b) Cleaning** | Drop duplicate rows, outcome and leaky columns, zero-guest bookings; normalise categories; ID columns → categorical. **Sensitivity checks** measure what the split strategy, deduplication and leaky features each do to the score | Implausible bedroom counts; transaction price anomalies judged against the **same postcode × type × bedrooms and quarter**, with thresholds learned on training years (a postcode-blind rule would have deleted 683 genuine luxury sales); irregular dates → weekly series |
| **Imputation** | Rule-based fills plus an **empirical comparison** of global median, group median and **KNN** for price | **KNN classifier** for bedrooms; empirical comparison of four methods for missing weeks, restricted to **causal** ones, because the filled values feed lag features |
| **Anomaly elimination** | IQR (shown to be too aggressive) vs **Isolation Forest** (removed, train only) | Modified z-score (transactions) + Isolation Forest cross-check; **robust STL residuals** (weekly series, train only) |
| **Clustering** | K-Means guest segments (elbow + silhouette) → new feature | K-Means postcodes by price-growth trajectory → cluster-level lag features |
| **Feature engineering** | Stay length, party size, price per person, booking history, top-N country and agent, … | Relative lags (1–52 weeks), momentum, volatility, volume, unit-price and cluster signals, calendar |
| **Feature selection** | Quasi-constant, correlation redundancy, mutual information vs a **random-probe noise floor** | **Permutation importance over time-ordered CV folds** vs a random-probe noise floor; the feature set (all / above noise / top 5 / top 10) is **chosen by time-ordered CV error** |
| **c) AutoML** | AutoViML × 4 model families (linear, random forest, XGBoost, CatBoost) + ensemble members vs scikit-learn baselines | AutoViML × 4 model families + ensemble members vs naive forecasting baselines, then a **2013–2019 walk-forward** with **Diebold-Mariano tests** (Holm-adjusted) and block-bootstrap confidence intervals |

Leakage is prevented throughout. Everything learned from data (imputers, scalers, anomaly detectors, clusters, feature selection) is fit on the **training split only**. **Both notebooks use a chronological split** (train on the past, test on later data), and the time-series features only use information from earlier weeks.

### Headline results

All numbers are on held-out test data the models never saw.

**Hotel cancellations** (17,544 deduplicated test bookings arriving 2 May – 31 Aug 2017; trained on earlier arrivals)

| Model | Balanced accuracy | ROC AUC |
|---|---|---|
| Majority-class baseline | 0.500 | n/a |
| Logistic regression baseline | 0.677 | 0.769 |
| AutoViML Linear | 0.647 | 0.719 |
| AutoViML Random Forest | 0.688 | 0.755 |
| AutoViML XGBoost | 0.702 | **0.777** |
| **AutoViML CatBoost** | **0.702** | 0.777 |

These numbers are deliberately lower than a naive setup. On the training period, the sensitivity checks show that a random split with duplicates kept scores **0.955 ROC AUC**, against **0.856** for the honest chronological split. Removing the leaky parking and suspect country features costs another 0.04. The test period is also harder: its cancellation rate is 34.7%, against 25.7% in training.

**Weekly median house price.** Holdout: 110 observed test weeks, May 2017 to Jul 2019.

| Model | Holdout MAE | Holdout MAPE |
|---|---|---|
| Naive (last week) | $36,129 | 5.43% |
| MA-13 (best naive baseline) | $30,597 | 4.58% |
| AutoViML XGBoost | $32,076 | 4.81% |
| AutoViML Linear | $31,516 | 4.79% |
| AutoViML Random Forest | $28,870 | 4.35% |
| **AutoViML CatBoost** | **$28,684** | **4.32%** |

**Walk-forward check.** Each year from 2013 to 2019 (331 weeks), the models are retrained on all earlier weeks and forecast that year. Only **AutoViML Random Forest** beats MA-13 significantly: its MAE is $1,547 lower (95% CI $535–$2,762, Holm-adjusted p = 0.021). The CatBoost gain ($985) is not significant after correction, Linear ties with MA-13, and XGBoost is significantly *worse*.

_TODO (you): add your interpretation._

### Running the notebooks

**Colab (recommended):** click a badge above. The first cell installs the pinned packages and restarts the runtime once. Then choose *Runtime → Run all*. Data is loaded from this repo automatically.

**Locally (Python 3.12):**

```bash
python -m venv .venv && source .venv/bin/activate
```

```bash
pip install -r part1/requirements.txt
```

```bash
jupyter lab
```

> **Version pins matter.** AutoViML 0.2.2 breaks with scikit-learn ≥ 1.4, IPython ≥ 9, NumPy 2 and setuptools ≥ 81. See [`part1/requirements.txt`](part1/requirements.txt) for details.
> Also note that importing AutoViz switches matplotlib to a non-interactive backend, so the notebooks re-enable `%matplotlib inline` right after importing it.

### Repository layout

```
data/
  hotel_bookings.csv
  property_sales/raw_sales.csv, ma_lga_12345.csv
  student_grade_ses_dataset.csv           # Part 2
part1/
  requirements.txt
  tabular_hotel_bookings/hotel_bookings_eda_prep_automl.ipynb
  timeseries_property_sales/property_sales_eda_prep_automl.ipynb
part2/
  requirements.txt                        # separate env from Part 1
  student_agentic_eda.ipynb
guidelines/                               # assignment instructions
```

### Use of AI tools

_TODO (you): describe how a coding agent was used to build these notebooks and what you reviewed, changed and wrote yourself (the assignment requires this to be explained clearly)._

---

## Part 2: Agentic EDA

| | |
|---|---|
| **Tool** | [business-science/ai-data-science-team](https://github.com/business-science/ai-data-science-team) (from the course slides): specialised LangChain/LangGraph agents that take natural-language instructions, choose tools or write and run pandas/plotly code, and explain the results |
| **Dataset** | [`student_grade_ses_dataset.csv`](data/student_grade_ses_dataset.csv): 12,025 students × 29 columns (socio-economic status, study habits, attendance, grades) |
| **Question** | What drives student performance, and which students are at risk? |
| **Notebook** | [`student_agentic_eda.ipynb`](part2/student_agentic_eda.ipynb) · [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anita-agasaveeran/CMPE255_Exploratory_Data_Analysis/blob/main/part2/student_agentic_eda.ipynb) |

**Agent workflow:** EDA Tools Agent (overview, statistics, missingness, Sweetviz) → Data Cleaning Agent (writes and runs a cleaning function) → EDA Tools Agent on the cleaned data (correlation funnel for at-risk students) → Data Visualization Agent → Pandas Data Analyst (multi-agent wrangling + charts) → Feature Engineering Agent. The notebook then **independently re-checks the agents' claims with plain pandas**, shows a run log of every prompt with its runtime and token cost, and compares agentic EDA with the manual EDA from Part 1.

**Running it:** an LLM is required. The default is OpenAI `gpt-4o-mini` (the library's default): add `OPENAI_API_KEY` as a Colab secret, or export it locally. A free local model via Ollama also works. Without a key, the notebook still runs its non-agent parts. Install into a **separate environment** from Part 1 with `pip install -r part2/requirements.txt`.

**Library bugs patched in the notebook:** version 0.0.0.9017 has three bugs that break agents in normal use:

- Agents given `user_instructions` hit an `AttributeError`.
- The visualisation agent's retry path hits an `re.error`.
- If LLM-written code calls `print()`, the library pastes the agent's full output table (millions of tokens) into its fix-the-code prompt, and OpenAI rejects it with a 429 *request too large* error.

All three are patched at runtime in section 0.1, with an explanation.
