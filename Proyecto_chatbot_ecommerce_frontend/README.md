# React Frontend – E-commerce Multimodal Chat UI

UI tipo chatbot (solo e-commerce) para tu backend local (FastAPI).

## Requisitos
- Node.js 18+

## Correr
1) Instalar deps:
   npm install

2) Levantar UI:
   npm run dev

Abre la URL que te muestra Vite (normalmente http://localhost:5173).

## Backend esperado
- POST http://127.0.0.1:8000/search/text  (JSON: {query, k})
- POST http://127.0.0.1:8000/search/image?k=10 (multipart form-data: file)

Este proyecto incluye proxy en `vite.config.js`, así que la UI llama:
- /search/text
- /search/image
