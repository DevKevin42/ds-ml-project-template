"""
Módulo para carga, exploración, limpieza y enriquecimiento
del dataset California Housing.

Este módulo:
1. Carga el dataset crudo.
2. Genera un resumen exploratorio básico.
3. Limpia inconsistencias y valores faltantes.
4. Crea variables derivadas.
5. Codifica variables categóricas.
6. Permite exportar el dataset final para modelado.
"""

from pathlib import Path
import pandas as pd


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """
    Carga el dataset desde un archivo CSV.

    Parámetros
    ----------
    csv_path : str | Path
        Ruta al archivo CSV.

    Retorna
    -------
    pd.DataFrame
        DataFrame cargado.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def explore_data(df: pd.DataFrame) -> dict:
    """
    Genera un resumen exploratorio básico del dataset.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame original.

    Retorna
    -------
    dict
        Diccionario con hallazgos clave del EDA.
    """
    if df.empty:
        raise ValueError("El DataFrame está vacío.")

    resumen_nulos = pd.DataFrame({
        "valores_nulos": df.isnull().sum(),
        "porcentaje_nulos": ((df.isnull().sum() / len(df)) * 100).round(4)
    }).sort_values(by="porcentaje_nulos", ascending=False)

    inconsistencias_bedrooms = df[
        df["total_bedrooms"].notna() &
        (df["total_bedrooms"] > df["total_rooms"])
    ]

    valor_max_precio = df["median_house_value"].max()

    hallazgos = {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "null_summary": resumen_nulos,
        "total_bedrooms_null_count": int(df["total_bedrooms"].isnull().sum()),
        "inconsistent_bedrooms_rows": int(len(inconsistencias_bedrooms)),
        "age_capped_count": int((df["housing_median_age"] == 52).sum()),
        "price_capped_count": int((df["median_house_value"] == valor_max_precio).sum()),
        "ocean_proximity_categories": sorted(df["ocean_proximity"].dropna().unique().tolist()),
        "numeric_description": df.describe(include="number")
    }

    return hallazgos


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Eliminar inconsistencias
    df = df[
        ~(
            df["total_bedrooms"].notna() &
            (df["total_bedrooms"] > df["total_rooms"])
        )
    ].copy()

    # Imputar faltantes
    mediana_bedrooms = df["total_bedrooms"].median()
    df["total_bedrooms"] = df["total_bedrooms"].fillna(mediana_bedrooms)

    # Flag válida
    df["is_age_capped"] = (df["housing_median_age"] >= 52).astype(int)

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["rooms_per_household"] = df["total_rooms"] / df["households"].replace(0, pd.NA)
    df["rooms_per_person"] = df["total_rooms"] / df["population"].replace(0, pd.NA)
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"].replace(0, pd.NA)

    df = pd.get_dummies(
        df,
        columns=["ocean_proximity"],
        drop_first=True
    )

    return df

def validate_processed_data(df: pd.DataFrame) -> dict:
    """
    Valida que el dataset final esté listo para modelado.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame ya procesado.

    Retorna
    -------
    dict
        Resumen de validación.
    """
    nulls_total = int(df.isnull().sum().sum())

    inconsistencias_restantes = 0
    if {"total_bedrooms", "total_rooms"}.issubset(df.columns):
        inconsistencias_restantes = int(
            (df["total_bedrooms"] > df["total_rooms"]).sum()
        )

    return {
        "shape": df.shape,
        "null_values_total": nulls_total,
        "remaining_inconsistencies_total_bedrooms_gt_total_rooms": inconsistencias_restantes,
        "columns": df.columns.tolist()
    }


def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orquesta limpieza y enriquecimiento del dataset.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame crudo.

    Retorna
    -------
    pd.DataFrame
        DataFrame procesado listo para modelado.
    """
    df_clean = clean_data(df)
    df_featured = create_features(df_clean)
    return df_featured


def save_processed_data(df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Guarda el DataFrame procesado en CSV.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame procesado.
    output_path : str | Path
        Ruta del archivo de salida.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    ruta_entrada = Path("../src/data/raw/housing/housing.csv")
    ruta_salida = Path("datos_preprocesados.csv")

    print("Cargando dataset crudo...")
    datos_crudos = load_data(ruta_entrada)

    print("Generando resumen exploratorio...")
    resumen_eda = explore_data(datos_crudos)
    print(f"Dimensiones del dataset: {resumen_eda['shape']}")
    print(f"Nulos en total_bedrooms: {resumen_eda['total_bedrooms_null_count']}")
    print(f"Filas inconsistentes total_bedrooms > total_rooms: {resumen_eda['inconsistent_bedrooms_rows']}")

    print("Aplicando limpieza y enriquecimiento...")
    datos_procesados = preprocess_pipeline(datos_crudos)

    print("Validando dataset final...")
    validacion = validate_processed_data(datos_procesados)
    print(f"Dimensiones finales: {validacion['shape']}")
    print(f"Valores nulos totales: {validacion['null_values_total']}")
    print(
        "Inconsistencias restantes total_bedrooms > total_rooms:",
        validacion["remaining_inconsistencies_total_bedrooms_gt_total_rooms"]
    )

    print(f"Guardando archivo procesado en: {ruta_salida}")
    save_processed_data(datos_procesados, ruta_salida)

    print("Proceso completado correctamente.")