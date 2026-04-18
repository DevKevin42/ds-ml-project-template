"""
API usando FastAPI para servir el modelo entrenado
de predicción de precios de vivienda en California.
"""

from pathlib import Path
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd


# =========================================================
# Configuración de rutas
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
FEATURES_PATH = BASE_DIR / "models" / "feature_columns.json"


# =========================================================
# Inicialización de la API
# =========================================================
app = FastAPI(
    title="API de Predicción de Precios de Vivienda (California)",
    version="1.0"
)


# =========================================================
# Esquema de entrada esperado por la API
# =========================================================
class HousingFeatures(BaseModel):
    longitude: float = Field(..., example=-122.23)
    latitude: float = Field(..., example=37.88)
    housing_median_age: float = Field(..., example=41.0)
    total_rooms: float = Field(..., example=880.0)
    total_bedrooms: float = Field(..., example=129.0)
    population: float = Field(..., example=322.0)
    households: float = Field(..., example=126.0)
    median_income: float = Field(..., example=8.3252)
    ocean_proximity: str = Field(..., example="NEAR BAY")


# =========================================================
# Variables globales
# =========================================================
model = None
feature_columns = None


# =========================================================
# Función de preprocesamiento
# =========================================================
def preprocess_input(data: HousingFeatures) -> pd.DataFrame:
    """
    Convierte la entrada del usuario en un DataFrame listo
    para ser consumido por el modelo, replicando la lógica
    del preprocesamiento utilizado en entrenamiento.
    """
    df = pd.DataFrame([data.model_dump()])

    # -----------------------------------------------------
    # Validaciones defensivas
    # -----------------------------------------------------
    numeric_columns = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income"
    ]

    for col in numeric_columns:
        if df[col].isnull().any():
            raise ValueError(f"La variable '{col}' no puede ser nula.")

    if df.loc[0, "households"] == 0:
        raise ValueError("La variable 'households' no puede ser 0.")

    if df.loc[0, "population"] == 0:
        raise ValueError("La variable 'population' no puede ser 0.")

    if df.loc[0, "total_rooms"] == 0:
        raise ValueError("La variable 'total_rooms' no puede ser 0.")

    if df.loc[0, "total_bedrooms"] > df.loc[0, "total_rooms"]:
        raise ValueError("La variable 'total_bedrooms' no puede ser mayor que 'total_rooms'.")

    # -----------------------------------------------------
    # Feature engineering
    # -----------------------------------------------------
    df["is_age_capped"] = (df["housing_median_age"] >= 52).astype(int)
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["rooms_per_person"] = df["total_rooms"] / df["population"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]

    # -----------------------------------------------------
    # Codificación categórica
    # -----------------------------------------------------
    df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

    # -----------------------------------------------------
    # Alinear columnas con las usadas durante entrenamiento
    # -----------------------------------------------------
    df = df.reindex(columns=feature_columns, fill_value=0)

    # -----------------------------------------------------
    # Validación final
    # -----------------------------------------------------
    total_nulls = int(df.isnull().sum().sum())
    if total_nulls > 0:
        raise ValueError("Existen valores nulos después del preprocesamiento.")

    return df


# =========================================================
# Evento de inicio
# =========================================================
@app.on_event("startup")
def load_artifacts() -> None:
    """
    Carga el modelo y las columnas esperadas al iniciar el servidor.
    """
    global model, feature_columns

    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")

        if not FEATURES_PATH.exists():
            raise FileNotFoundError(f"No se encontró el archivo de columnas en: {FEATURES_PATH}")

        model = joblib.load(MODEL_PATH)

        with open(FEATURES_PATH, "r", encoding="utf-8") as f:
            feature_columns = json.load(f)

        print("Modelo y columnas cargados correctamente.")

    except Exception as e:
        print(f"Advertencia: No se pudieron cargar los artefactos: {e}")
        model = None
        feature_columns = None


# =========================================================
# Endpoints
# =========================================================
@app.get("/")
def home():
    return {
        "mensaje": "Bienvenido a la API del Proyecto Final de Ciencia de Datos"
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None and feature_columns is not None else "error",
        "model_loaded": model is not None,
        "feature_columns_loaded": feature_columns is not None,
        "model_path": str(MODEL_PATH),
        "features_path": str(FEATURES_PATH)
    }


@app.post("/predict")
def predict_price(features: HousingFeatures):
    """
    Recibe variables de entrada, aplica preprocesamiento
    y retorna la predicción del precio estimado.
    """
    if model is None or feature_columns is None:
        raise HTTPException(
            status_code=500,
            detail="El modelo o las columnas de entrenamiento no se han cargado correctamente."
        )

    try:
        X = preprocess_input(features)
        prediction = model.predict(X)[0]

        return {
            "predicted_price": float(prediction),
            "currency": "USD"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))