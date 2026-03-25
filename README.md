# Life Expectancy Prediction — Africa Health AI

## Mission
To leverage AI and digital platforms to improve access to quality healthcare in Africa by enabling early disease detection, efficient patient management, and data-driven decision-making for underserved communities.

## Problem Statement
Predicting life expectancy in African nations using healthcare, immunization, and socioeconomic indicators to identify data-driven interventions for underserved communities.

## Dataset
- **Source**: [Kaggle — Life Expectancy (WHO)](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)
- **Size**: 2,938 rows × 22 columns
- **Coverage**: 193 countries across 16 years (2000–2015)
- **Features**: Immunization rates, mortality indicators, GDP, schooling, BMI, HIV/AIDS prevalence, and more
- **Target**: `Life expectancy` (continuous, in years)

---

## Task 1 — Jupyter Notebook

**File**: [`summative/linear_regression/multivariate.ipynb`](summative/linear_regression/multivariate.ipynb)

- Exploratory data analysis: correlation heatmap, feature distributions, scatter plots, outlier detection, missing value map — each with training decisions
- Feature engineering: dropped multicollinear columns (`under-five deaths`, `thinness 5-9 years`, `percentage expenditure`), encoded `Status`, log-transformed skewed features, median-imputed missing values
- Standardisation using `StandardScaler` (fit on training data only — no leakage)
- Three models trained and compared:

| Model | Test MSE | Test R² |
|-------|----------|---------|
| Linear Regression (OLS + SGD) | 14.02 | 0.839 |
| Decision Tree (CV-tuned depth) | 6.16 | 0.929 |
| **Random Forest (300 trees)** | **3.03** | **0.965** |

- Best model (Random Forest) saved as `best_model.pkl` → used by the API

---

## Task 2 — FastAPI Prediction Service

**File**: [`summative/API/prediction.py`](summative/API/prediction.py)

| | Link |
|--|------|
| **Base URL** | [https://life-expectancy-api-frsw.onrender.com](https://life-expectancy-api-frsw.onrender.com) |
| **Swagger UI** | [https://life-expectancy-api-frsw.onrender.com/docs](https://life-expectancy-api-frsw.onrender.com/docs) |

### Endpoints
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Predict life expectancy from 16 input features |
| `POST` | `/retrain` | Retrain the model with new labelled data rows |

### Features
- All 16 input variables validated with **Pydantic** — typed (`int`/`float`) with `ge`/`le` range constraints
- **CORS** configured with explicit allowed origins, methods, headers and credentials (no wildcard `*`)
- `/retrain` accepts new `(X, y)` rows, re-fits the RandomForestRegressor, saves the updated model to disk, and reloads it in-memory immediately — subsequent `/predict` calls use the new model

### Run locally
```bash
cd summative/API
pip install -r requirements.txt
uvicorn prediction:app --reload --port 8080
# Open http://127.0.0.1:8080/docs
```

---

## Task 3 — Flutter Mobile App

**Directory**: [`summative/FlutterApp/life_expectancy_app/`](summative/FlutterApp/life_expectancy_app/)

A single-page mobile app that calls the live API to predict life expectancy.

### Features
- **16 input text fields** — one per model feature, grouped by category (Mortality, Immunisation, Economic, Lifestyle & Social)
- **Development Status dropdown** — `Developing` / `Developed`
- **Predict button** — sticky bottom bar, always visible
- **Result display** — colour-coded card: green with progress bar on success, red on error, dashed border when empty
- Input validation with range constraints matching the API — errors shown inline per field
- Auto-scrolls to the result card after a successful prediction

### Run locally
```bash
cd summative/FlutterApp/life_expectancy_app
flutter pub get
flutter run
```

> **Note**: The app connects to the live Render API. On first request after inactivity the server may take up to 30 seconds to wake from sleep (Render free tier cold start).

---

## Video Demo
*Coming soon*

---

## Repository Structure

```
linear_regression_model/
│
├── summative/
│   ├── linear_regression/
│   │   └── multivariate.ipynb            # Task 1 — notebook
│   ├── API/                              # Task 2 — FastAPI service
│   │   ├── prediction.py                 # API app
│   │   ├── requirements.txt              # Python dependencies
│   │   ├── best_model.pkl                # RandomForestRegressor (300 trees)
│   │   ├── scaler.pkl                    # StandardScaler
│   │   ├── imputer.pkl                   # SimpleImputer (median)
│   │   └── feature_columns.pkl           # Ordered feature name list (16)
│   └── FlutterApp/
│       └── life_expectancy_app/          # Task 3 — Flutter mobile app
│           └── lib/main.dart             # Full app source
│
└── README.md
```
