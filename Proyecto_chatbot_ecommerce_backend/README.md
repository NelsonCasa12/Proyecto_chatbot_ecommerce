# Backend v2 – Multimodal (Texto + Imagen) con Datafiniti

Tus CSV tienen ~34k+ **filas** (reseñas). Eso NO significa 34k productos únicos.
Antes veías "70 productos" porque el script agregaba por ASIN y filtraba solo los que tenían imagen.

Esta versión genera:
- `data/corpus_reviews.parquet`: ~34k documentos (1 fila = 1 reseña) → sirve para cumplir "30,000 items"
- `data/products_agg.parquet`: 1 fila por ASIN (útil para UI/resumen)

Y crea 2 índices FAISS:
- `data/faiss_text.index` + `data/meta_text.npy`: indexa TODOS los docs (texto)
- `data/faiss_image.index` + `data/meta_image.npy`: indexa SOLO docs con imagen descargable (imagen)

## Pasos
1) pip install -r requirements.txt
2) Copia tus 3 CSV a `data/` con estos nombres:
   - 1429_1.csv
   - Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv
   - Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv
3) python -m src.prepare_data
4) python -m src.build_index
5) uvicorn src.api.main:app --reload --port 8000

## Endpoints
- POST /search/text  (busca sobre ~34k docs)
- POST /search/image (busca sobre docs con imagen válida)