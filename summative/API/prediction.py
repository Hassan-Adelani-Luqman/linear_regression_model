import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Load artifacts once at startup ────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model           = joblib.load(os.path.join(BASE_DIR, "best_model.pkl"))
scaler          = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
imputer         = joblib.load(os.path.join(BASE_DIR, "imputer.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Life Expectancy Prediction API",
    description=(
        "Predicts a country's life expectancy from 16 health, economic, and "
        "social indicators using a Random Forest model trained on WHO data (2000–2015). "
        "Also exposes a /retrain endpoint to update the model with new data."
    ),
    version="1.0.0",
)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:8080",
        "http://localhost:8100",
        "http://127.0.0.1:8000",
        "https://life-expectancy-api-frsw.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)

# ── Columns log1p-transformed during training (skewness > 1.0 after imputation) ─
# Determined by notebook Cell 3.6: skew_series[skew_series > 1.0].index.tolist()
LOG_COLS = [
    "Population", "infant deaths", "Measles",
    "HIV/AIDS", "GDP", "thinness 1-19 years", "Adult Mortality",
]

# ── Field name map: Pydantic identifier → exact feature_columns name ──────────
FIELD_MAP = {
    "Status"              : "Status",
    "Adult_Mortality"     : "Adult Mortality",
    "infant_deaths"       : "infant deaths",
    "Alcohol"             : "Alcohol",
    "Hepatitis_B"         : "Hepatitis B",
    "Measles"             : "Measles",
    "BMI"                 : "BMI",
    "Polio"               : "Polio",
    "Total_expenditure"   : "Total expenditure",
    "Diphtheria"          : "Diphtheria",
    "HIV_AIDS"            : "HIV/AIDS",
    "GDP"                 : "GDP",
    "Population"          : "Population",
    "thinness_1_19_years" : "thinness 1-19 years",
    "Income_composition"  : "Income composition of resources",
    "Schooling"           : "Schooling",
}

# ── Pydantic models ────────────────────────────────────────────────────────────
class PredictionInput(BaseModel):
    Status: int = Field(
        ..., ge=0, le=1,
        description="Development status — 0 = Developed, 1 = Developing"
    )
    Adult_Mortality: float = Field(
        ..., ge=1.0, le=800.0,
        description="Adult mortality rate per 1,000 population (ages 15–60)"
    )
    infant_deaths: float = Field(
        ..., ge=0.0, le=1800.0,
        description="Number of infant deaths per 1,000 population"
    )
    Alcohol: float = Field(
        ..., ge=0.0, le=20.0,
        description="Alcohol consumption — litres of pure alcohol per capita"
    )
    Hepatitis_B: float = Field(
        ..., ge=1.0, le=100.0,
        description="Hepatitis B immunisation coverage among 1-year-olds (%)"
    )
    Measles: float = Field(
        ..., ge=0.0, le=250000.0,
        description="Number of reported Measles cases per 1,000 population"
    )
    BMI: float = Field(
        ..., ge=1.0, le=90.0,
        description="Average Body Mass Index of the entire population"
    )
    Polio: float = Field(
        ..., ge=1.0, le=100.0,
        description="Polio immunisation coverage among 1-year-olds (%)"
    )
    Total_expenditure: float = Field(
        ..., ge=0.0, le=20.0,
        description="Government health expenditure as % of total government expenditure"
    )
    Diphtheria: float = Field(
        ..., ge=1.0, le=100.0,
        description="DTP3 immunisation coverage among 1-year-olds (%)"
    )
    HIV_AIDS: float = Field(
        ..., ge=0.1, le=50.0,
        description="Deaths per 1,000 live births caused by HIV/AIDS (0–4 year olds)"
    )
    GDP: float = Field(
        ..., ge=1.0, le=120000.0,
        description="GDP per capita (USD)"
    )
    Population: float = Field(
        ..., ge=34.0, le=1_400_000_000.0,
        description="Population of the country"
    )
    thinness_1_19_years: float = Field(
        ..., ge=0.1, le=30.0,
        description="Prevalence of thinness among children and adolescents aged 10–19 (%)"
    )
    Income_composition: float = Field(
        ..., ge=0.0, le=1.0,
        description="Human Development Index income composition of resources (0–1 scale)"
    )
    Schooling: float = Field(
        ..., ge=0.0, le=21.0,
        description="Average number of years of schooling"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "Status": 1,
                "Adult_Mortality": 263,
                "infant_deaths": 62,
                "Alcohol": 0.01,
                "Hepatitis_B": 65,
                "Measles": 1154,
                "BMI": 19.1,
                "Polio": 6,
                "Total_expenditure": 8.16,
                "Diphtheria": 65,
                "HIV_AIDS": 0.1,
                "GDP": 584.26,
                "Population": 33736494,
                "thinness_1_19_years": 17.2,
                "Income_composition": 0.479,
                "Schooling": 10.1,
            }
        }
    }


class RetrainInput(BaseModel):
    data: list[PredictionInput] = Field(
        ..., min_length=1,
        description="List of new training rows (one PredictionInput per row)"
    )
    targets: list[float] = Field(
        ..., min_length=1,
        description="Corresponding life expectancy values (years) for each row"
    )


# ── Helper: convert PredictionInput → correctly ordered numpy array ────────────
def input_to_array(item: PredictionInput) -> pd.DataFrame:
    raw = {FIELD_MAP[k]: v for k, v in item.model_dump().items()}
    df  = pd.DataFrame([raw])[feature_columns]          # enforce column order
    df_imputed = pd.DataFrame(
        imputer.transform(df), columns=feature_columns
    )
    # Mirror the training pipeline: log1p-transform skewed features before scaling
    for col in LOG_COLS:
        df_imputed[col] = np.log1p(df_imputed[col])
    df_scaled  = scaler.transform(df_imputed)
    return df_scaled


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status" : "ok",
        "message": "Life Expectancy Prediction API is running.",
        "docs"   : "/docs",
    }


@app.post("/predict", tags=["Prediction"])
def predict(input_data: PredictionInput):
    """
    Predict life expectancy (years) for a single country record.

    Supply all 16 features. The model applies the same imputation,
    log-transform, and standardisation pipeline used during training.
    Returns the predicted life expectancy in years.
    """
    try:
        X = input_to_array(input_data)
        prediction = float(model.predict(X)[0])
        return {
            "predicted_life_expectancy_years": round(prediction, 2),
            "model": type(model).__name__,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain", tags=["Retraining"])
def retrain(payload: RetrainInput):
    """
    Retrain the model with new labelled data.

    Send a list of PredictionInput rows and corresponding target values.
    The API re-fits a fresh RandomForestRegressor on the supplied data,
    overwrites best_model.pkl, and updates the in-memory model immediately.
    Subsequent /predict calls use the retrained model.
    """
    global model

    if len(payload.data) != len(payload.targets):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Length mismatch: {len(payload.data)} data rows "
                f"but {len(payload.targets)} targets."
            ),
        )

    try:
        # Build feature matrix
        rows = []
        for item in payload.data:
            raw = {FIELD_MAP[k]: v for k, v in item.model_dump().items()}
            rows.append(raw)

        X_df      = pd.DataFrame(rows)[feature_columns]
        X_imputed = pd.DataFrame(
            imputer.transform(X_df), columns=feature_columns
        )
        X_scaled  = scaler.transform(X_imputed)
        y         = np.array(payload.targets)

        # Re-fit — clamp split/leaf params to dataset size so small
        # payloads don't crash (sklearn raises if min_samples > n_samples)
        from sklearn.ensemble import RandomForestRegressor as RFR
        from sklearn.metrics  import mean_squared_error, r2_score

        n = len(y)
        new_model = RFR(
            n_estimators=300,
            max_depth=15,
            min_samples_split=min(5, max(2, n)),
            min_samples_leaf=min(2, max(1, n // 2)),
            random_state=42,
            n_jobs=1,
        )
        new_model.fit(X_scaled, y)

        # Persist to disk and update in-memory reference
        model_path = os.path.join(BASE_DIR, "best_model.pkl")
        joblib.dump(new_model, model_path)
        model = new_model

        # Report fit quality on the training batch
        # r2_score returns NaN when n=1 (SS_tot=0); replace with 1.0 (perfect fit)
        y_pred  = new_model.predict(X_scaled)
        r2      = r2_score(y, y_pred)
        r2_safe = 1.0 if (r2 != r2) else round(float(r2), 4)   # NaN check
        return {
            "status"         : "retrained",
            "rows_used"      : len(y),
            "train_mse"      : round(float(mean_squared_error(y, y_pred)), 4),
            "train_r2"       : r2_safe,
            "model_saved_to" : model_path,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
