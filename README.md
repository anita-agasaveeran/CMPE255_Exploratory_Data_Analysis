# CMPE 255: Exploratory Data Analysis, Data Preparation & AutoML

Assignment for CMPE 255 (Data Mining), SJSU, Fall 2026.

| Part | What | Status |
|---|---|---|
| **Part 1** | Manual + automated EDA, data cleaning and preparation, clustering, anomaly removal, imputation, feature engineering and selection, and AutoML (AutoViML) on a **tabular** and a **time-series** Kaggle dataset | ✅ notebooks executed |
| **Part 2** | EDA done by an **AI data-analysis agent** instead of by hand | ⏳ to do |

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
| **a) EDA and visualisation** | Schema and missingness audit, duplicates, impossible values, **target-leakage check**, distributions, cancellation drivers, seasonality, countries, Spearman correlations | Coverage over time, price distributions, trend and volume, sales seasonality, postcode × year heatmap, STL decomposition, ADF stationarity tests, ACF/PACF, cross-check against the publisher's moving average |
| **Auto-EDA** | AutoViz | AutoViz |
| **b) Cleaning** | Drop 27% duplicate rows, leakage columns, zero-guest bookings; normalise categories; ID columns → categorical | Implausible bedroom counts; transaction price anomalies (modified z-score within peer groups); irregular dates → weekly series |
| **Imputation** | Rule-based fills plus an **empirical comparison** of global median, group median and **KNN** for price | **KNN classifier** for bedrooms; empirical comparison of forward-fill, linear and rolling-median for missing weeks |
| **Anomaly elimination** | IQR (shown to be too aggressive) vs **Isolation Forest** (removed, train only) | Modified z-score (transactions) + Isolation Forest cross-check; **robust STL residuals** (weekly series, train only) |
| **Clustering** | K-Means guest segments (elbow + silhouette) → new feature | K-Means postcodes by price-growth trajectory → cluster-level lag features |
| **Feature engineering** | Stay length, party size, price per person, booking history, top-N country and agent, … | Relative lags (1–52 weeks), momentum, volatility, volume, unit-price and cluster signals, calendar |
| **Feature selection** | Quasi-constant, correlation redundancy, mutual information vs a **random-probe noise floor** | Correlation redundancy, mutual information vs random-probe noise floor |
| **c) AutoML** | AutoViML × 4 model families (linear, random forest, XGBoost, CatBoost) + ensemble members vs scikit-learn baselines | AutoViML × 4 model families + ensemble members vs naive forecasting baselines |

Leakage is prevented throughout. Everything learned from data (imputers, scalers, anomaly detectors, clusters, feature selection) is fit on the **training split only**. The time series uses a **chronological** split, and its features only use information from earlier weeks.

### Headline results

All numbers are on held-out test data the models never saw.

**Hotel cancellations** (17,446 test bookings after deduplication)

| Model | Balanced accuracy | ROC AUC |
|---|---|---|
| Majority-class baseline | 0.500 | n/a |
| Logistic regression baseline | 0.728 | 0.861 |
| AutoViML Linear | 0.717 | 0.791 |
| AutoViML Random Forest | 0.792 | 0.878 |
| AutoViML XGBoost | 0.819 | **0.909** |
| **AutoViML CatBoost** | **0.819** | 0.907 |

**Weekly median house price** (114 test weeks, May 2017 to Jul 2019)

| Model | MAE | MAPE |
|---|---|---|
| Naive (last week) | $33,016 | 5.01% |
| MA-13 (best naive baseline) | $28,873 | 4.36% |
| AutoViML Random Forest (best single AutoViML model) | $27,123 | 4.12% |
| **AutoViML Linear → ensemble** | **$26,629** | **4.06%** |

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
  student_grade_ses_dataset.csv           # reserved for Part 2
part1/
  requirements.txt
  tabular_hotel_bookings/hotel_bookings_eda_prep_automl.ipynb
  timeseries_property_sales/property_sales_eda_prep_automl.ipynb
part2/                                    # agentic EDA (to do)
guidelines/                               # assignment instructions
```

### Use of AI tools

_TODO (you): describe how a coding agent was used to build these notebooks and what you reviewed, changed and wrote yourself (the assignment requires this to be explained clearly)._

---

## Part 2: Agentic EDA

_To do. Planned dataset: [`student_grade_ses_dataset.csv`](data/student_grade_ses_dataset.csv), analysed with one of the agentic data-analysis tools from the course slides._
