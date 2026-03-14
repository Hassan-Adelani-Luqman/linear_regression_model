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

## Repository Structure

```
linear_regression_model/
│
├── summative/
│   ├── linear_regression/
│   │   └── multivariate.ipynb       # Main notebook
│   ├── API/                         # Saved model artifacts
│   └── FlutterApp/                  # Mobile app (coming soon)
│
└── README.md
```

## Notebook Overview (`multivariate.ipynb`)
- Exploratory data analysis: correlation heatmap, feature distributions, scatter plots, outlier detection, missing value map — each with training decisions
- Feature engineering: dropped multicollinear columns (`under-five deaths`, `thinness 5-9 years`, `percentage expenditure`), encoded `Status`, log-transformed skewed features, median-imputed missing values
- Standardization using `StandardScaler` (fit on training data only — no leakage)
- Three models: Linear Regression (OLS + SGDRegressor gradient descent), Decision Tree (CV-tuned depth), Random Forest (300 trees)
- Loss curves (train vs test MSE per epoch), before/after scatter plots, model comparison table
- Best-performing model (Random Forest — Test MSE: 3.03, R²: 0.965) saved as `best_model.pkl`
