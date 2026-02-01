import re

ECOMMERCE_KEYWORDS = [
    "buy", "price", "cheap", "recommend", "review", "rating", "amazon", "product",
    "headphones", "boots", "shoes", "speaker", "tablet", "kindle", "echo", "fire",
    "wireless", "bluetooth", "camera", "charger", "keyboard", "mouse", "monitor",
    # spanish keywords (helps scope filter)
    "audifonos", "audífonos", "botas", "zapatos", "parlante", "tablet", "teclado", "raton", "mouse"
]

def is_ecommerce_query(q: str) -> bool:
    q = (q or "").lower()
    if not q.strip():
        return False
    return any(k in q for k in ECOMMERCE_KEYWORDS)

def safe_refusal() -> str:
    return (
        "Puedo ayudarte solo con preguntas de e-commerce (productos, recomendaciones, "
        "comparaciones, reseñas, precios aproximados). "
        "Intenta con algo como: 'audífonos inalámbricos baratos' o 'botas negras'."
    )
