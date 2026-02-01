"""
build_index.py

Este script construye un índice vectorial (FAISS)
a partir de los embeddings multimodales de productos.

Entrada:
- data/processed/products_embeddings.pkl

Salida:
- data/index/faiss_index.bin
- data/index/index_metadata.pkl

Responsabilidades:
- cargar embeddings
- construir índice vectorial
- guardar índice y metadata

Este script NO:
- procesa consultas
- hace re-ranking
- genera texto

Autor: Nelson Casa
Materia: Recuperación de la Información
"""


import os
import pickle
import numpy as np
import faiss

# Rutas del proyecto

# --------------------

# Ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Entrada
EMBEDDINGS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "products_embeddings.pkl"
)

# Carpeta de salida del índice
INDEX_DIR = os.path.join(
    BASE_DIR,
    "data",
    "index"
)

os.makedirs(INDEX_DIR, exist_ok=True)

# Archivos de salida
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(INDEX_DIR, "index_metadata.pkl")


# ---------------------------
# Cargar embeddings

def load_embeddings(path):
    """
    Carga los embeddings multimodales desde disco.
    """
    print("Cargando embeddings...")
    with open(path, "rb") as f:
        data = pickle.load(f)

    print(f"Embeddings cargados: {len(data)} productos")
    return data


# ---------------------------
# Preparar vectores para FAISS

def prepare_vectors(embeddings_data):
    """
    Extrae los embeddings de texto y prepara la matriz para FAISS.
    """
    print("Preparando vectores para indexación...")

    vectors = []
    metadata = []

    for item in embeddings_data:
        vectors.append(item["text_embedding"])
        metadata.append({
            "product_id": item["product_id"],
            "name": item["metadata"]["name"],
            "brand": item["metadata"]["brand"],
            "categories": item["metadata"]["categories"],
            "image_url": item["metadata"]["image_url"]
        })

    vectors = np.array(vectors).astype("float32")

    print(f"Matriz de vectores: {vectors.shape}")
    return vectors, metadata


# ---------------------------
# Construir índice FAISS
""" Usaremos IndexFlatL2:
 * simple
 * exacto
 * perfecto para datasets pequeños/medianos
""" 

def build_faiss_index(vectors):
    """
    Construye un índice FAISS usando distancia L2.
    """
    dim = vectors.shape[1]
    print(f"Construyendo índice FAISS (dim={dim})...")

    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    print(f"Índice construido con {index.ntotal} vectores")
    return index


# ---------------------------
# Guardar índice y metadata

def save_index(index, metadata):
    """
    Guarda el índice FAISS y la metadata asociada.
    """
    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print("Índice y metadata guardados correctamente")


# Función principal

def main():
    # 1. Cargar embeddings
    embeddings_data = load_embeddings(EMBEDDINGS_PATH)

    # 2. Preparar vectores y metadata
    vectors, metadata = prepare_vectors(embeddings_data)

    # 3. Construir índice FAISS
    index = build_faiss_index(vectors)

    # 4. Guardar índice
    save_index(index, metadata)


if __name__ == "__main__":
    main()
