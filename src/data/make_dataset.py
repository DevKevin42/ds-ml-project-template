"""
Script para descargar y extraer el dataset de California Housing.

Flujo:
1. Descarga el archivo housing.tgz desde la URL indicada.
2. Lo guarda en src/data/raw/housing/
3. Extrae su contenido en esa misma carpeta.
"""

from pathlib import Path
import urllib.request
import tarfile


def fetch_housing_data(housing_url: str, housing_path: str) -> Path:
    """
    Descarga un archivo .tgz desde una URL y extrae su contenido
    en el directorio indicado.

    Parámetros
    ----------
    housing_url : str
        URL del archivo .tgz.
    housing_path : str
        Ruta del directorio donde se guardará y extraerá el contenido.

    Retorna
    -------
    Path
        Ruta del archivo CSV extraído esperado.
    """
    housing_dir = Path(housing_path)
    housing_dir.mkdir(parents=True, exist_ok=True)

    tgz_path = housing_dir / "housing.tgz"

    print(f"Descargando dataset desde: {housing_url}")
    urllib.request.urlretrieve(housing_url, tgz_path)

    print(f"Archivo descargado en: {tgz_path}")
    print("Extrayendo contenido...")

    with tarfile.open(tgz_path, mode="r:gz") as housing_tgz:
        housing_tgz.extractall(path=housing_dir)

    csv_path = housing_dir / "housing.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo esperado después de extraer: {csv_path}"
        )

    print(f"Dataset extraído correctamente en: {csv_path}")
    return csv_path


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]
    HOUSING_URL = "https://github.com/ageron/data/raw/main/housing.tgz"
    HOUSING_PATH = BASE_DIR / "data" / "raw" / "housing"

    print("Iniciando descarga y extracción de datos...")
    fetch_housing_data(HOUSING_URL, str(HOUSING_PATH))
    print("Datos descargados y extraídos correctamente.")