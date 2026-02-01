"""
embed_products.py

Este script genera embeddings multimodales (texto + imagen)
para los productos del corpus procesado.

Entrada:
- data/processed/products_corpus.csv

Salida:
- data/processed/products_embeddings.pkl

Responsabilidades:
- cargar el corpus limpio
- descargar imágenes de productos
- generar embeddings de texto e imagen usando CLIP
- almacenar embeddings junto con metadata del producto

Este script NO:
- crea índices vectoriales
- realiza búsquedas
- hace re-ranking

Autor: Nelson Casa
Materia: Recuperación de la Información
"""


# Imports necesarios

import os
import pickle
import requests
from io import BytesIO

import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Rutas del proyecto

# Ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Archivos de entrada
PROCESSED_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "products_corpus.csv"
)

# Archivo de salida
EMBEDDINGS_OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "products_embeddings.pkl"
)


"""
Cargar el modelo CLIP
Usaremos CLIP porque:

* es multimodal
* texto e imagen comparten espacio vectorial
* es el estándar académico
"""

def load_clip_model():
    """
    Carga el modelo CLIP y su procesador.
    """
    print("Cargando modelo CLIP...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor

# Descargar imagen del producto 
# Las imágenes vienen como URLs

def load_image_from_url(url):
    """
    Descarga una imagen desde una URL y la carga como objeto PIL.
    """
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image
    except Exception as e:
        print(f"Error cargando imagen: {e}")
        return None




# Generar embeddings de un producto

def generate_embeddings(model, processor, text, image):
    """
    Genera embeddings de texto e imagen usando CLIP.
    El texto se trunca automáticamente a 77 tokens.
    """
    inputs = processor(
        text=[text],
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )

    with torch.no_grad():
        outputs = model(**inputs)

    text_embedding = outputs.text_embeds[0].cpu().numpy()
    image_embedding = outputs.image_embeds[0].cpu().numpy()

    return text_embedding, image_embedding


# Procesar todo el corpus

def process_corpus_embeddings(df, model, processor):
    """
    Recorre el corpus y genera embeddings para cada producto.
    """
    print("Generando embeddings para los productos...")

    embeddings_data = []

    for idx, row in df.iterrows():
        print(f"Procesando producto {idx + 1}/{len(df)}")

        product_id = row["asins"]
        product_text = row["product_text"] 
        image_urls = row["imageURLs"]
 
        # Tomamos solo la primera imagen
        image_url = image_urls.split(",")[0] if image_urls else None
        image = load_image_from_url(image_url) if image_url else None

        if image is None:
            print("Imagen no disponible, se omite producto")
            continue

        text_emb, image_emb = generate_embeddings(
            model, processor, product_text, image
        )

        embeddings_data.append({
            "product_id": product_id,
            "text_embedding": text_emb,
            "image_embedding": image_emb,
            "metadata": {
                "name": row["name"],
                "brand": row["brand"],
                "categories": row["categories"],
                "image_url": image_url
            }
        })

    return embeddings_data

# Nota:
# El texto se trunca para cumplir con la limitación de tokens de CLIP.
# El texto completo se conserva para la etapa de RAG.



# Guardar embeddings

def save_embeddings(data, path):
    """
    Guarda los embeddings en un archivo pickle.
    """
    with open(path, "wb") as f:
        pickle.dump(data, f)

    print(f"Embeddings guardados en {path}")


# Función principal

def main():
    # 1. Cargar corpus procesado
    print("Cargando corpus procesado...")
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # 2. Cargar modelo CLIP
    model, processor = load_clip_model()

    # 3. Generar embeddings
    embeddings = process_corpus_embeddings(df, model, processor)

    # 4. Guardar embeddings
    save_embeddings(embeddings, EMBEDDINGS_OUTPUT_PATH)

if __name__ == "__main__":
    main()
