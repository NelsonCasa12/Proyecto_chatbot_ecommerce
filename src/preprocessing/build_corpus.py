"""
build_corpus.py

Este script se encarga de construir el corpus final de productos
a partir del archivo crudo:
Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv

Responsabilidades de este script:
- Cargar el dataset crudo
- Analizar columnas disponibles
- Definir la unidad documental como PRODUCTO
- Agrupar reseñas por producto
- Construir un texto consolidado por producto
- Guardar el corpus limpio en data/processed/

Este script NO:
- genera embeddings
- crea índices
- realiza búsquedas

Autor: Nelson Casa
Materia: Recuperación de la Información
"""

import pandas as pd
import os

# Ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Rutas de datos
RAW_DATA_PATHS = [
    os.path.join(
        BASE_DIR,
        "data",
        "raw",
        "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"
    ),
    os.path.join(
        BASE_DIR,
        "data",
        "raw",
        "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv"
    )
]


PROCESSED_DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

# Crear carpeta processed si no existe
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)



# Cargar el dataset crudo

def load_raw_data(paths):
    """
    Carga y unifica múltiples archivos CSV multimodales.
    """
    dfs = []
    for path in paths:
        print(f"Cargando {os.path.basename(path)}")
        df = pd.read_csv(path)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Dataset combinado con {combined_df.shape[0]} filas")
    return combined_df




# Inspección inicial de columnas 

def inspect_columns(df):
    """
    Muestra información básica del dataset para análisis exploratorio.
    """
    print("\nColumnas disponibles en el dataset:\n")
    for col in df.columns:
        print(f"- {col}")


def select_relevant_columns(df):
    """
    Selecciona las columnas relevantes para construir el corpus de productos.
    """
    selected_columns = [
        "asins",
        "name",
        "brand",
        "categories",
        "reviews.text",
        "reviews.title",
        "imageURLs"
    ]

    # Filtrar solo columnas que existen realmente
    existing_columns = [col for col in selected_columns if col in df.columns]

    print("\nColumnas seleccionadas:")
    for col in existing_columns:
        print(f"- {col}")

    return df[existing_columns]


# Limpieza básica

def basic_cleaning(df):
    """
    Limpieza básica del dataset:
    - eliminar filas sin identificador de producto
    - rellenar valores nulos con texto vacío
    """
    print("\nRealizando limpieza básica...")

    df = df.dropna(subset=["asins"])
    df = df.fillna("")

    print(f"Filas restantes después de limpieza: {df.shape[0]}")
    return df

def normalize_asins(df):
    """
    Normaliza la columna ASINs:
    - separa múltiples ASINs en filas independientes
    """
    print("\nNormalizando ASINs (separando múltiples identificadores)...")

    df["asins"] = df["asins"].str.split(",")
    df = df.explode("asins")
    df["asins"] = df["asins"].str.strip()

    print(f"Filas después de normalizar ASINs: {df.shape[0]}")
    return df


# Agrupar por productos

def build_product_corpus(df):
    """
    Agrupa las filas por producto (ASIN) y construye
    un documento por producto.
    """
    print("\nConstruyendo corpus por producto...")

    grouped = df.groupby("asins").agg({
        "name": "first",
        "brand": "first",
        "categories": "first",
        "imageURLs": "first",
        "reviews.title": lambda x: " ".join(x),
        "reviews.text": lambda x: " ".join(x)
    }).reset_index()

    print(f"Productos únicos en el corpus: {grouped.shape[0]}")
    return grouped


# Construir el texto final del producto

def build_text_field(df):
    """
    Construye el texto consolidado del producto
    que se usará para embeddings.
    """
    print("\nConstruyendo campo de texto consolidado...")

    df["product_text"] = (
        "Product name: " + df["name"] + ". " +
        "Brand: " + df["brand"] + ". " +
        "Categories: " + df["categories"] + ". " +
        "Review titles: " + df["reviews.title"] + ". " +
        "Reviews: " + df["reviews.text"]
    )

    return df


# Guardar el corpus procesado

def save_processed_corpus(df):
    """
    Guarda el corpus procesado en formato CSV y JSON.
    """
    csv_path = os.path.join(PROCESSED_DATA_DIR, "products_corpus.csv")
    json_path = os.path.join(PROCESSED_DATA_DIR, "products_corpus.json")

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", lines=True)

    print(f"\nCorpus guardado en:")
    print(f"- {csv_path}")
    print(f"- {json_path}")


# Función principal

def main():
    # 1. Cargar y unificar los CSV multimodales
    df_raw = load_raw_data(RAW_DATA_PATHS)

    # 2. Inspección inicial del corpus
    inspect_columns(df_raw)

    # 3. Selección de columnas relevantes
    df_selected = select_relevant_columns(df_raw)

    # 4. Limpieza básica
    df_clean = basic_cleaning(df_selected)

    # 5. Normalización de ASINs (separar múltiples IDs)
    df_normalized = normalize_asins(df_clean)

    # 6. Construcción del corpus por producto
    df_products = build_product_corpus(df_normalized)

    # 7. Construcción del texto consolidado
    df_final = build_text_field(df_products)

    # 8. Guardar el corpus procesado
    save_processed_corpus(df_final)



if __name__ == "__main__":
    main()
