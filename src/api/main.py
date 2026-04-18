"""
API con FastAPI para servir el modelo entrenado
y ofrecer una interfaz web amigable para el usuario.
"""

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import joblib
import pandas as pd

# =========================
# Inicialización de la app
# =========================
app = FastAPI(
    title="API de Predicción de Precios de Vivienda (California)",
    version="2.0"
)

# =========================
# Esquema de entrada JSON
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
    ocean_proximity: str

# =========================
# Variable global del modelo
# =========================
model = None

# =========================
# Cargar modelo al iniciar
# =========================
@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load("models/best_model.pkl")
        print("Modelo cargado correctamente.")
    except Exception as e:
        print("Error cargando modelo:", e)

# =========================
# Preprocesamiento
# =========================
def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Replica la lógica de transformación utilizada en entrenamiento.
    """
    df = pd.DataFrame([data])

    # Evitar divisiones por cero
    if df.loc[0, "households"] == 0:
        raise ValueError("La variable 'households' no puede ser 0.")
    if df.loc[0, "population"] == 0:
        raise ValueError("La variable 'population' no puede ser 0.")
    if df.loc[0, "total_rooms"] == 0:
        raise ValueError("La variable 'total_rooms' no puede ser 0.")

    # Feature engineering
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["rooms_per_person"] = df["total_rooms"] / df["population"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]

    # One-Hot Encoding manual
    categorias = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]

    for cat in categorias:
        df[f"ocean_proximity_{cat}"] = 1 if data["ocean_proximity"] == cat else 0

    # Eliminar columna categórica original
    df = df.drop(columns=["ocean_proximity"])

    # Asegurar orden exacto de columnas que espera el modelo
    if hasattr(model, "feature_names_in_"):
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    return df

# =========================
# HTML base reutilizable
# =========================
def render_page(resultado: str = "", error: str = "") -> str:
    return f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Predicción de Precios de Vivienda</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #eef2f7, #d9e7ff);
                margin: 0;
                padding: 0;
            }}
            .container {{
                max-width: 850px;
                margin: 40px auto;
                background: white;
                border-radius: 16px;
                box-shadow: 0 8px 24px rgba(0,0,0,0.12);
                padding: 30px;
            }}
            h1 {{
                text-align: center;
                color: #1f3c88;
                margin-bottom: 10px;
            }}
            p {{
                text-align: center;
                color: #444;
                margin-bottom: 25px;
            }}
            form {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 16px;
            }}
            .full {{
                grid-column: 1 / -1;
            }}
            label {{
                font-weight: bold;
                display: block;
                margin-bottom: 6px;
                color: #222;
            }}
            input, select {{
                width: 100%;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 10px;
                font-size: 14px;
                box-sizing: border-box;
            }}
            button {{
                grid-column: 1 / -1;
                padding: 14px;
                background: #1f3c88;
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
            }}
            button:hover {{
                background: #16306d;
            }}
            .resultado {{
                margin-top: 25px;
                padding: 18px;
                border-radius: 12px;
                background: #eaf7ea;
                color: #14532d;
                font-size: 18px;
                font-weight: bold;
                text-align: center;
            }}
            .error {{
                margin-top: 25px;
                padding: 18px;
                border-radius: 12px;
                background: #fdecec;
                color: #9f1239;
                font-size: 16px;
                text-align: center;
                font-weight: bold;
            }}
            .nota {{
                margin-top: 20px;
                font-size: 13px;
                color: #666;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Predicción de Precio de Vivienda</h1>
            <p>Ingresa las características de la vivienda para estimar su precio.</p>

            <form action="/predict-form" method="post">
                <div>
                    <label>Longitud</label>
                    <input type="number" step="any" name="longitude" required>
                </div>

                <div>
                    <label>Latitud</label>
                    <input type="number" step="any" name="latitude" required>
                </div>

                <div>
                    <label>Edad media de la vivienda</label>
                    <input type="number" step="any" name="housing_median_age" required>
                </div>

                <div>
                    <label>Total de habitaciones</label>
                    <input type="number" step="any" name="total_rooms" required>
                </div>

                <div>
                    <label>Total de dormitorios</label>
                    <input type="number" step="any" name="total_bedrooms" required>
                </div>

                <div>
                    <label>Población</label>
                    <input type="number" step="any" name="population" required>
                </div>

                <div>
                    <label>Hogares</label>
                    <input type="number" step="any" name="households" required>
                </div>

                <div>
                    <label>Ingreso medio</label>
                    <input type="number" step="any" name="median_income" required>
                </div>

                <div class="full">
                    <label>Proximidad al océano</label>
                    <select name="ocean_proximity" required>
                        <option value="<1H OCEAN">&lt;1H OCEAN</option>
                        <option value="INLAND">INLAND</option>
                        <option value="ISLAND">ISLAND</option>
                        <option value="NEAR BAY">NEAR BAY</option>
                        <option value="NEAR OCEAN">NEAR OCEAN</option>
                    </select>
                </div>

                <button type="submit">Predecir precio</button>
            </form>

            {"<div class='resultado'>" + resultado + "</div>" if resultado else ""}
            {"<div class='error'>" + error + "</div>" if error else ""}

            <div class="nota">
                También puedes probar la API en formato técnico desde <strong>/docs</strong>.
            </div>
        </div>
    </body>
    </html>
    """

# =========================
# Página principal amigable
# =========================
@app.get("/", response_class=HTMLResponse)
def home():
    return render_page()

# =========================
# Endpoint JSON técnico
# =========================
@app.post("/predict")
def predict_price(features: HousingFeatures):
    if model is None:
        return JSONResponse(
            status_code=500,
            content={"error": "El modelo no se ha cargado correctamente."}
        )

    try:
        data = features.dict()
        df = preprocess_input(data)
        prediction = model.predict(df)[0]

        return {
            "predicted_price": float(prediction),
            "message": "Predicción exitosa"
        }

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )

# =========================
# Endpoint desde formulario
# =========================
@app.post("/predict-form", response_class=HTMLResponse)
def predict_form(
    longitude: float = Form(...),
    latitude: float = Form(...),
    housing_median_age: float = Form(...),
    total_rooms: float = Form(...),
    total_bedrooms: float = Form(...),
    population: float = Form(...),
    households: float = Form(...),
    median_income: float = Form(...),
    ocean_proximity: str = Form(...)
):
    if model is None:
        return render_page(error="El modelo no se ha cargado correctamente.")

    try:
        data = {
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income,
            "ocean_proximity": ocean_proximity
        }

        df = preprocess_input(data)
        prediction = model.predict(df)[0]

        resultado = f"Precio estimado de la vivienda: ${prediction:,.2f}"
        return render_page(resultado=resultado)

    except Exception as e:
        return render_page(error=f"Ocurrió un error: {str(e)}")