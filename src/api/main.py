"""
API robusta con FastAPI para servir un modelo de predicción de precios
de vivienda en California, incluyendo una interfaz web amigable.
"""

from typing import Optional
from html import escape

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
import joblib
import pandas as pd


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
app = FastAPI(
    title="Predicción de Precios de Vivienda (California)",
    version="3.0"
)

MODEL_PATH = "models/best_model.pkl"

# Columnas esperadas por el modelo entrenado
EXPECTED_COLUMNS = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "rooms_per_household",
    "rooms_per_person",
    "bedrooms_per_room",
    "ocean_proximity_<1H OCEAN",
    "ocean_proximity_INLAND",
    "ocean_proximity_ISLAND",
    "ocean_proximity_NEAR BAY",
    "ocean_proximity_NEAR OCEAN",
]

VALID_OCEAN_PROXIMITY = {
    "<1H OCEAN",
    "INLAND",
    "ISLAND",
    "NEAR BAY",
    "NEAR OCEAN",
}

# Modelo global
model = None


# =========================================================
# ESQUEMA JSON DE ENTRADA
# =========================================================
class HousingFeatures(BaseModel):
    longitude: float = Field(..., ge=-180, le=180, description="Longitud geográfica")
    latitude: float = Field(..., ge=-90, le=90, description="Latitud geográfica")
    housing_median_age: float = Field(..., ge=1, le=100, description="Edad media de la vivienda")
    total_rooms: float = Field(..., gt=0, description="Total de habitaciones")
    total_bedrooms: float = Field(..., gt=0, description="Total de dormitorios")
    population: float = Field(..., gt=0, description="Población")
    households: float = Field(..., gt=0, description="Número de hogares")
    median_income: float = Field(..., gt=0, description="Ingreso medio")
    ocean_proximity: str = Field(..., description="Proximidad al océano")

    @field_validator("ocean_proximity")
    @classmethod
    def validate_ocean_proximity(cls, value: str) -> str:
        value = value.strip().upper()
        if value not in VALID_OCEAN_PROXIMITY:
            raise ValueError(
                "ocean_proximity debe ser uno de estos valores: "
                "<1H OCEAN, INLAND, ISLAND, NEAR BAY, NEAR OCEAN"
            )
        return value

    @field_validator("total_bedrooms")
    @classmethod
    def validate_bedrooms_vs_rooms(cls, value: float, info):
        total_rooms = info.data.get("total_rooms")
        if total_rooms is not None and value > total_rooms:
            raise ValueError("total_bedrooms no puede ser mayor que total_rooms.")
        return value


# =========================================================
# CARGA DEL MODELO
# =========================================================
@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print("Modelo cargado correctamente.")
    except Exception as e:
        model = None
        print(f"Error cargando modelo: {e}")


# =========================================================
# PREPROCESAMIENTO
# =========================================================
def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Replica la lógica de transformación del notebook de limpieza
    y genera exactamente las columnas esperadas por el modelo.
    """
    df = pd.DataFrame([data])

    # Validaciones defensivas extra
    if df.loc[0, "households"] <= 0:
        raise ValueError("La variable 'households' debe ser mayor que 0.")
    if df.loc[0, "population"] <= 0:
        raise ValueError("La variable 'population' debe ser mayor que 0.")
    if df.loc[0, "total_rooms"] <= 0:
        raise ValueError("La variable 'total_rooms' debe ser mayor que 0.")
    if df.loc[0, "total_bedrooms"] > df.loc[0, "total_rooms"]:
        raise ValueError("total_bedrooms no puede ser mayor que total_rooms.")

    # Variables derivadas
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["rooms_per_person"] = df["total_rooms"] / df["population"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]

    # One-hot encoding manual
    for category in VALID_OCEAN_PROXIMITY:
        col_name = f"ocean_proximity_{category}"
        df[col_name] = 1 if data["ocean_proximity"] == category else 0

    # Eliminar columna categórica original
    df.drop(columns=["ocean_proximity"], inplace=True)

    # Reordenar y asegurar columnas faltantes
    df = df.reindex(columns=EXPECTED_COLUMNS, fill_value=0)

    return df


# =========================================================
# RENDER HTML
# =========================================================
def render_page(
    resultado: str = "",
    error: str = "",
    values: Optional[dict] = None
) -> str:
    values = values or {}

    def v(key: str, default: str = "") -> str:
        return escape(str(values.get(key, default)))

    resultado_html = f"<div class='resultado'>{escape(resultado)}</div>" if resultado else ""
    error_html = f"<div class='error'>{escape(error)}</div>" if error else ""

    return f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Predicción de Precio de Vivienda</title>
        <style>
            * {{
                box-sizing: border-box;
            }}
            body {{
                margin: 0;
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #edf3ff, #dbeafe);
                color: #1f2937;
            }}
            .wrapper {{
                max-width: 980px;
                margin: 30px auto;
                padding: 20px;
            }}
            .card {{
                background: white;
                border-radius: 20px;
                box-shadow: 0 12px 30px rgba(0,0,0,0.12);
                overflow: hidden;
            }}
            .header {{
                background: #1e3a8a;
                color: white;
                padding: 28px;
            }}
            .header h1 {{
                margin: 0 0 8px 0;
                font-size: 30px;
            }}
            .header p {{
                margin: 0;
                color: #dbeafe;
                line-height: 1.5;
            }}
            .content {{
                padding: 28px;
            }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 18px;
            }}
            .full {{
                grid-column: 1 / -1;
            }}
            label {{
                display: block;
                font-weight: bold;
                margin-bottom: 8px;
                color: #111827;
            }}
            input, select {{
                width: 100%;
                padding: 12px;
                border: 1px solid #cbd5e1;
                border-radius: 12px;
                font-size: 15px;
                background: #f8fafc;
            }}
            input:focus, select:focus {{
                outline: none;
                border-color: #2563eb;
                box-shadow: 0 0 0 4px rgba(37,99,235,0.15);
            }}
            .hint {{
                margin-top: 6px;
                font-size: 12px;
                color: #64748b;
            }}
            .actions {{
                margin-top: 24px;
                display: flex;
                gap: 12px;
                flex-wrap: wrap;
            }}
            button {{
                border: none;
                background: #2563eb;
                color: white;
                padding: 14px 20px;
                border-radius: 12px;
                font-weight: bold;
                font-size: 15px;
                cursor: pointer;
            }}
            button:hover {{
                background: #1d4ed8;
            }}
            .secondary {{
                background: #e5e7eb;
                color: #111827;
                text-decoration: none;
                padding: 14px 20px;
                border-radius: 12px;
                font-weight: bold;
                font-size: 15px;
            }}
            .resultado {{
                margin-top: 24px;
                background: #ecfdf5;
                border: 1px solid #a7f3d0;
                color: #065f46;
                padding: 18px;
                border-radius: 14px;
                font-size: 20px;
                font-weight: bold;
                text-align: center;
            }}
            .error {{
                margin-top: 24px;
                background: #fef2f2;
                border: 1px solid #fecaca;
                color: #991b1b;
                padding: 18px;
                border-radius: 14px;
                font-size: 16px;
                font-weight: bold;
                text-align: center;
            }}
            .footer-note {{
                margin-top: 24px;
                font-size: 13px;
                color: #475569;
                line-height: 1.6;
                background: #f8fafc;
                border-radius: 12px;
                padding: 16px;
            }}
            code {{
                background: #e2e8f0;
                padding: 2px 6px;
                border-radius: 6px;
            }}
            @media (max-width: 768px) {{
                .grid {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="wrapper">
            <div class="card">
                <div class="header">
                    <h1>Predicción de Precio de Vivienda</h1>
                    <p>
                        Ingresa las características de una vivienda en California y obtén una
                        estimación del precio usando el modelo entrenado del proyecto.
                    </p>
                </div>

                <div class="content">
                    <form action="/predict-form" method="post">
                        <div class="grid">
                            <div>
                                <label for="longitude">Longitud</label>
                                <input id="longitude" name="longitude" type="number" step="any" min="-180" max="180" value="{v('longitude')}" required>
                                <div class="hint">Ejemplo: -122.23</div>
                            </div>

                            <div>
                                <label for="latitude">Latitud</label>
                                <input id="latitude" name="latitude" type="number" step="any" min="-90" max="90" value="{v('latitude')}" required>
                                <div class="hint">Ejemplo: 37.88</div>
                            </div>

                            <div>
                                <label for="housing_median_age">Edad media de la vivienda</label>
                                <input id="housing_median_age" name="housing_median_age" type="number" step="any" min="1" max="100" value="{v('housing_median_age')}" required>
                                <div class="hint">Rango razonable: 1 a 52+</div>
                            </div>

                            <div>
                                <label for="median_income">Ingreso medio</label>
                                <input id="median_income" name="median_income" type="number" step="any" min="0.1" value="{v('median_income')}" required>
                                <div class="hint">Ejemplo: 8.3252</div>
                            </div>

                            <div>
                                <label for="total_rooms">Total de habitaciones</label>
                                <input id="total_rooms" name="total_rooms" type="number" step="any" min="1" value="{v('total_rooms')}" required>
                            </div>

                            <div>
                                <label for="total_bedrooms">Total de dormitorios</label>
                                <input id="total_bedrooms" name="total_bedrooms" type="number" step="any" min="1" value="{v('total_bedrooms')}" required>
                                <div class="hint">No puede ser mayor que total_rooms</div>
                            </div>

                            <div>
                                <label for="population">Población</label>
                                <input id="population" name="population" type="number" step="any" min="1" value="{v('population')}" required>
                            </div>

                            <div>
                                <label for="households">Hogares</label>
                                <input id="households" name="households" type="number" step="any" min="1" value="{v('households')}" required>
                            </div>

                            <div class="full">
                                <label for="ocean_proximity">Proximidad al océano</label>
                                <select id="ocean_proximity" name="ocean_proximity" required>
                                    <option value="">Selecciona una opción</option>
                                    <option value="<1H OCEAN" {"selected" if v("ocean_proximity") == "<1H OCEAN" else ""}>&lt;1H OCEAN</option>
                                    <option value="INLAND" {"selected" if v("ocean_proximity") == "INLAND" else ""}>INLAND</option>
                                    <option value="ISLAND" {"selected" if v("ocean_proximity") == "ISLAND" else ""}>ISLAND</option>
                                    <option value="NEAR BAY" {"selected" if v("ocean_proximity") == "NEAR BAY" else ""}>NEAR BAY</option>
                                    <option value="NEAR OCEAN" {"selected" if v("ocean_proximity") == "NEAR OCEAN" else ""}>NEAR OCEAN</option>
                                </select>
                            </div>
                        </div>

                        <div class="actions">
                            <button type="submit">Evaluar vivienda</button>
                            <a class="secondary" href="/">Limpiar formulario</a>
                            <a class="secondary" href="/docs" target="_blank">Probar API técnica</a>
                        </div>
                    </form>

                    {resultado_html}
                    {error_html}

                    <div class="footer-note">
                        <strong>Nota:</strong> esta interfaz utiliza el mismo modelo del proyecto.
                        También puedes enviar solicitudes JSON al endpoint <code>/predict</code>.
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """


# =========================================================
# PÁGINA PRINCIPAL
# =========================================================
@app.get("/", response_class=HTMLResponse)
def home():
    return render_page()


# =========================================================
# ENDPOINT JSON
# =========================================================
@app.post("/predict")
def predict_price(features: HousingFeatures):
    if model is None:
        return JSONResponse(
            status_code=500,
            content={"error": "El modelo no se ha cargado correctamente."}
        )

    try:
        data = features.model_dump()
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


# =========================================================
# ENDPOINT FORMULARIO
# =========================================================
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
        return render_page(
            error="El modelo no se ha cargado correctamente.",
            values={
                "longitude": longitude,
                "latitude": latitude,
                "housing_median_age": housing_median_age,
                "total_rooms": total_rooms,
                "total_bedrooms": total_bedrooms,
                "population": population,
                "households": households,
                "median_income": median_income,
                "ocean_proximity": ocean_proximity,
            }
        )

    try:
        data = HousingFeatures(
            longitude=longitude,
            latitude=latitude,
            housing_median_age=housing_median_age,
            total_rooms=total_rooms,
            total_bedrooms=total_bedrooms,
            population=population,
            households=households,
            median_income=median_income,
            ocean_proximity=ocean_proximity,
        ).model_dump()

        df = preprocess_input(data)
        prediction = model.predict(df)[0]

        resultado = f"Precio estimado de la vivienda: ${prediction:,.2f}"

        return render_page(
            resultado=resultado,
            values=data
        )

    except Exception as e:
        return render_page(
            error=f"Ocurrió un error: {str(e)}",
            values={
                "longitude": longitude,
                "latitude": latitude,
                "housing_median_age": housing_median_age,
                "total_rooms": total_rooms,
                "total_bedrooms": total_bedrooms,
                "population": population,
                "households": households,
                "median_income": median_income,
                "ocean_proximity": ocean_proximity,
            }
        )