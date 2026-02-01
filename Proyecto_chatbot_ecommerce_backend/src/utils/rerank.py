from __future__ import annotations
from typing import List, Dict, Any
import math
import re

def _tokenize(s: str) -> List[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return [t for t in s.split() if len(t) > 2]

def rerank(query: str, candidates: List[Dict[str, Any]], top_k: int = 10, prefs: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    prefs = prefs or {}
    q_tokens = set(_tokenize(query))

    def score_item(item: Dict[str, Any]) -> float:
        text = " ".join([str(item.get("name","")), str(item.get("brand","")), str(item.get("categories","")), str(item.get("review_text",""))]).lower()
        t_tokens = set(_tokenize(text))
        overlap = len(q_tokens & t_tokens)

        # Base: semantic score from FAISS if present
        base = float(item.get("score", 0.0))

        # Rating boost
        rating = item.get("rating")
        if rating is None:
            rating = item.get("avg_rating")
        try:
            rating = float(rating) if rating is not None else None
        except:
            rating = None
        rating_boost = 0.0 if rating is None else (rating / 5.0) * 0.10

        # Pref boosts
        pref_boost = 0.0
        if prefs.get("category_hint"):
            if prefs["category_hint"] in text:
                pref_boost += 0.08
        if prefs.get("color"):
            if str(prefs["color"]).lower() in text:
                pref_boost += 0.05

        # Lexical overlap scaled
        lex = min(0.20, overlap * 0.02)

        return base + lex + rating_boost + pref_boost

    ranked = sorted(candidates, key=score_item, reverse=True)
    return ranked[:top_k]
