# Importa la clase Path para trabajar con rutas de forma moderna y legible
from pathlib import Path

# Importa urllib.request para descargar archivos desde una URL
import urllib.request

# Importa tarfile para abrir y extraer archivos comprimidos .tgz o .tar.gz
import tarfile


# Define una función que recibe la URL del archivo y la ruta de destino
def fetch_housing_data(housing_url: str, housing_path: str) -> None:
    """
    Descarga un archivo .tgz desde una URL y extrae su contenido
    en el directorio indicado.

    Parámetros:
        housing_url (str): URL del archivo .tgz
        housing_path (str): ruta del directorio donde se guardará y extraerá
    """

    # Convierte la ruta recibida en un objeto Path para manipularla mejor
    housing_dir = Path(housing_path)

    # Crea la carpeta destino y, si no existen carpetas intermedias, también las crea
    housing_dir.mkdir(parents=True, exist_ok=True)

    # Define la ruta completa del archivo comprimido que se descargará
    tgz_path = housing_dir / "housing.tgz"

    # Descarga el archivo desde la URL y lo guarda en la ruta indicada
    urllib.request.urlretrieve(housing_url, tgz_path)

    # Abre el archivo comprimido .tgz en modo lectura
    with tarfile.open(tgz_path, mode="r:gz") as housing_tgz:

        # Extrae todo el contenido del archivo comprimido en la carpeta destino
        housing_tgz.extractall(path=housing_dir)


# Verifica si este archivo se está ejecutando directamente y no siendo importado
if __name__ == "__main__":

    # Guarda la URL del dataset en una variable para reutilizarla fácilmente
    housing_url = "https://github.com/ageron/data/raw/main/housing.tgz"

    # Define la carpeta donde se descargará y extraerá el archivo
    housing_path = "./ds-ml-project-template/src/data/raw"

    # Imprime un mensaje indicando que el proceso de descarga comenzará
    print("Iniciando descarga y extracción de datos...")

    # Llama a la función principal para descargar y extraer el dataset
    fetch_housing_data(housing_url, housing_path)

    # Imprime un mensaje confirmando que el proceso terminó correctamente
    print("Datos descargados y extraídos correctamente.")