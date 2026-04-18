"""
Script para dividir los datos en conjunto de entrenamiento y conjunto de prueba.

Estrategia:
1. Carga el dataset crudo.
2. Genera una variable categórica de apoyo basada en median_income
   para realizar una división estratificada.
3. Divide los datos en train y test.
4. Guarda ambos archivos en la carpeta interim.

Nota:
La limpieza y el feature engineering deben aplicarse después,
idealmente ajustando transformaciones sobre train y replicándolas en test.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def split_and_save_data(raw_data_path: str, interim_data_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lee el dataset crudo, lo divide en entrenamiento y prueba usando
    estratificación basada en median_income, y guarda ambos archivos CSV.

    Parámetros
    ----------
    raw_data_path : str
        Ruta del archivo CSV crudo.
    interim_data_path : str
        Carpeta donde se guardarán train_set.csv y test_set.csv.

    Retorna
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        train_set y test_set.
    """
    raw_path = Path(raw_data_path)
    interim_path = Path(interim_data_path)

    # =========================
    # Validaciones iniciales
    # =========================
    if not raw_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {raw_path}")

    interim_path.mkdir(parents=True, exist_ok=True)

    # =========================
    # Cargar datos
    # =========================
    df = pd.read_csv(raw_path)

    if df.empty:
        raise ValueError("El archivo CSV está vacío.")

    if "median_income" not in df.columns:
        raise ValueError("La columna 'median_income' no existe en el dataset.")

    # =========================
    # Crear variable de estratificación
    # =========================
    # Se limita a 5 categorías como en el enfoque clásico del proyecto California Housing
    df = df.copy()
    df["income_cat"] = pd.cut(
        df["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, float("inf")],
        labels=[1, 2, 3, 4, 5]
    )

    # =========================
    # División estratificada
    # =========================
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in splitter.split(df, df["income_cat"]):
        train_set = df.loc[train_index].copy()
        test_set = df.loc[test_index].copy()

    # Eliminar columna auxiliar antes de guardar
    train_set.drop(columns=["income_cat"], inplace=True)
    test_set.drop(columns=["income_cat"], inplace=True)

    # =========================
    # Guardar archivos
    # =========================
    train_output_path = interim_path / "train_set.csv"
    test_output_path = interim_path / "test_set.csv"

    train_set.to_csv(train_output_path, index=False)
    test_set.to_csv(test_output_path, index=False)

    # =========================
    # Resumen por consola
    # =========================
    print("División completada correctamente.")
    print(f"Dataset original: {df.shape}")
    print(f"Train set: {train_set.shape} -> {train_output_path}")
    print(f"Test set: {test_set.shape} -> {test_output_path}")

    return train_set, test_set


if __name__ == "__main__":
    RAW_PATH = "src/data/raw/housing/housing.csv"
    INTERIM_PATH = "src/data/interim/"

    split_and_save_data(RAW_PATH, INTERIM_PATH)
