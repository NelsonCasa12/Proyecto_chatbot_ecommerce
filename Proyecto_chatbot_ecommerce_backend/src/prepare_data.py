import os
import re
import pandas as pd

DATA_DIR = "data"
OUT_REVIEWS = os.path.join(DATA_DIR, "corpus_reviews.parquet")
OUT_PRODUCTS = os.path.join(DATA_DIR, "products_agg.parquet")

CSV_FILES = [
    os.path.join(DATA_DIR, "1429_1.csv"),
    os.path.join(DATA_DIR, "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv"),
    os.path.join(DATA_DIR, "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"),
]

KEEP_COLS = ["asins","name","brand","categories","imageURLs","reviews.text","reviews.rating","reviews.date"]

def clean_text(t: str) -> str:
    t = "" if t is None else str(t)
    return re.sub(r"\s+", " ", t).strip()

def first_image(urls):
    if urls is None or (isinstance(urls, float) and pd.isna(urls)):
        return None
    parts = [p.strip() for p in str(urls).split(",") if p.strip()]
    for p in parts:
        if p.startswith("http://") or p.startswith("https://"):
            return p
    return None

def normalize_asins(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    s = str(val).strip()
    s = s.strip("[]").replace("'", "").replace('"',"")
    parts = [p.strip() for p in s.split(",") if p.strip() and p.strip().lower() != "nan"]
    return parts

def main():
    dfs = []
    for path in CSV_FILES:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No existe: {path}")
        print(f"Leyendo: {path}")
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        for c in KEEP_COLS:
            if c not in df.columns:
                df[c] = None
        dfs.append(df[KEEP_COLS].copy())

    df = pd.concat(dfs, ignore_index=True)
    print("Filas totales (reseñas):", len(df))

    df["reviews.text"] = df["reviews.text"].apply(clean_text)
    df["name"] = df["name"].apply(clean_text)
    df["brand"] = df["brand"].apply(clean_text)
    df["categories"] = df["categories"].apply(clean_text)
    df["image_url"] = df["imageURLs"].apply(first_image)
    df["reviews.rating"] = pd.to_numeric(df["reviews.rating"], errors="coerce")

    # No exigir imagen aquí, para conservar corpus grande
    df = df[df["reviews.text"].str.len() > 10]
    df = df[df["name"].str.len() > 1]

    df["asin_list"] = df["asins"].apply(normalize_asins)
    df = df[df["asin_list"].map(len) > 0]

    df = df.explode("asin_list").rename(columns={"asin_list":"asin"})
    df["asin"] = df["asin"].astype(str).str.strip()

    corpus = df[["asin","name","brand","categories","image_url","reviews.text","reviews.rating","reviews.date"]].copy()
    corpus = corpus.rename(columns={"reviews.text":"review_text","reviews.rating":"rating","reviews.date":"date"})
    corpus["text_for_embed"] = (corpus["name"] + " " + corpus["brand"] + " " + corpus["categories"] + " " + corpus["review_text"]).apply(clean_text)

    corpus.to_parquet(OUT_REVIEWS, index=False)
    print("Corpus (docs por reseña) guardado:", len(corpus), "->", OUT_REVIEWS)

    products = (
        corpus.groupby("asin", as_index=False)
        .agg(
            name=("name","first"),
            brand=("brand","first"),
            categories=("categories","first"),
            image_url=("image_url","first"),
            avg_rating=("rating","mean"),
            num_reviews=("review_text","count"),
            reviews_joined=("review_text", lambda x: " ".join(x.dropna().astype(str).tolist()[:25]))
        )
    )
    products.to_parquet(OUT_PRODUCTS, index=False)
    print("Productos únicos (ASIN):", len(products), "->", OUT_PRODUCTS)

if __name__ == "__main__":
    main()
