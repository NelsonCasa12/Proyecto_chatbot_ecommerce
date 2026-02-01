"""
search.py

Este módulo implementa la recuperación inicial (retrieval)
de productos a partir de una consulta textual.

Utiliza:
- embeddings CLIP (texto)
- índice vectorial FAISS

Entrada:
- consulta de texto del usuario

Salida:
- lista de productos candidatos (top-k)

Autor: Nelson Casa
Materia: Recuperación de la Información
"""

# Imports necesarios
import os
import pickle
import numpy as np
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel


# -----------------------------------
# Rutas del proyecto

# Ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Archivos del índice
FAISS_INDEX_PATH = os.path.join(
    BASE_DIR, "data", "index", "faiss_index.bin"
)

METADATA_PATH = os.path.join(
    BASE_DIR, "data", "index", "index_metadata.pkl"
)


# -----------------------------------
# Cargar índice y metadata

def load_index_and_metadata():
    """
    Carga el índice FAISS y la metadata asociada.
    """
    print("Cargando índice FAISS y metadata...")

    index = faiss.read_index(FAISS_INDEX_PATH)

    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    print(f"Índice cargado con {index.ntotal} productos")
    return index, metadata

# -----------------------------------
# Cargar modelo CLIP

def load_clip_model():
    """
    Carga CLIP para generar embeddings de consultas textuales.
    """
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor

# -----------------------------------
# Generar embedding de la consulta

def embed_query(text_query, model, processor):
    """
    Genera embedding de texto para la consulta del usuario
    usando CLIP en modo texto-only.
    Devuelve un vector (1, dim) compatible con FAISS.
    """
    inputs = processor(
        text=[text_query],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)

    # Convertir a numpy
    query_embedding = text_features.cpu().numpy().astype("float32")

    # Asegurar forma (1, dim)
    query_embedding = query_embedding.reshape(1, -1)

    return query_embedding


# -----------------------------------
# Búsqueda en el índice (top-k)

def search_products(query_embedding, index, metadata, top_k=5):
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        product_info = metadata[idx]
        results.append({
            "rank": rank + 1,
            "distance": float(distances[0][rank]),
            "product_id": product_info["product_id"],
            "name": product_info["name"],
            "brand": product_info["brand"],
            "categories": product_info["categories"],
            "image_url": product_info["image_url"]
        })

    return results



# -----------------------------------
# Función principal

def main():
    # 1. Cargar índice y metadata
    index, metadata = load_index_and_metadata()

    # 2. Cargar CLIP
    model, processor = load_clip_model()

    # 3. Consulta de prueba
    query = input("\nIngresa una consulta de texto: ")

    # 4. Generar embedding de la consulta
    query_embedding = embed_query(query, model, processor)

    # 5. Buscar productos
    results = search_products(query_embedding, index, metadata, top_k=5)

    # 6. Mostrar resultados
    print("\nResultados de la búsqueda:\n")
    for res in results:
        print(f"Rank {res['rank']}")
        print(f"Producto: {res['name']}")
        print(f"Marca: {res['brand']}")
        print(f"Categoría: {res['categories']}")
        print(f"Distancia: {res['distance']:.4f}")
        print("-" * 40)

if __name__ == "__main__":
    main()
