"""
API Básica usando FastAPI para servir el modelo entrenado.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Inicializamos la app
app = FastAPI(
    title="API de Predicción de Precios de Viviendcleara (California)",
    version="1.0"
)

# =========================
# INPUT DEL USUARIO
# =========================
class HousingFeatures(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str  # importante para encoding

# =========================
# CARGA DEL MODELO
# =========================
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load("models/best_model.pkl")
        print("Modelo cargado correctamente.")
    except Exception as e:
        print("Error cargando modelo:", e)

# =========================
# ENDPOINTS
# =========================
@app.get("/")
def home():
    return {"mensaje": "API de predicción funcionando"}

# =========================
# FUNCIÓN DE TRANSFORMACIÓN
# =========================
def preprocess_input(data: dict):
    """
    Replica la lógica del Notebook 2
    """

    df = pd.DataFrame([data])

    # =====================
    # FEATURE ENGINEERING
    # =====================
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["rooms_per_person"] = df["total_rooms"] / df["population"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]

    # =====================
    # ONE HOT ENCODING
    # =====================
    categorias = [
        "<1H OCEAN",
        "INLAND",
        "ISLAND",
        "NEAR BAY",
        "NEAR OCEAN"
    ]

    for cat in categorias:
        df[f"ocean_proximity_{cat}"] = 1 if data["ocean_proximity"] == cat else 0

    # Elimina columna original
    df = df.drop(columns=["ocean_proximity"])

    return df

# =========================
# PREDICCIÓN
# =========================
@app.post("/predict")
def predict_price(features: HousingFeatures):

    if model is None:
        return {"error": "Modelo no cargado"}

    try:
        # Convertir input a dict
        data = features.dict()

        # Preprocesar igual que entrenamiento
        df = preprocess_input(data)

        # Asegurar orden de columnas (CRÍTICO)
        model_columns = model.feature_names_in_
        df = df[model_columns]

        # Predicción
        prediction = model.predict(df)[0]

        return {
            "predicted_price": float(prediction),
            "message": "Predicción exitosa"
        }

    except Exception as e:
        return {"error": str(e)}