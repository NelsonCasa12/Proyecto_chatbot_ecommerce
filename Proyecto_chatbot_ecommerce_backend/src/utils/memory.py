from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any
import time
import re

@dataclass
class SessionMemory:
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())
    prefs: Dict[str, Any] = field(default_factory=dict)
    last_queries: List[str] = field(default_factory=list)
    last_results: List[Dict[str, Any]] = field(default_factory=list)

class MemoryStore:
    def __init__(self, ttl_seconds: int = 60*60*6):
        self.ttl = ttl_seconds
        self._store: Dict[str, SessionMemory] = {}

    def get(self, session_id: str) -> SessionMemory:
        self._gc()
        if session_id not in self._store:
            self._store[session_id] = SessionMemory()
        return self._store[session_id]

    def update_query(self, session_id: str, q: str):
        m = self.get(session_id)
        m.updated_at = time.time()
        m.last_queries = (m.last_queries + [q])[-10:]

    def update_results(self, session_id: str, results):
        m = self.get(session_id)
        m.updated_at = time.time()
        m.last_results = results[:20]

    def update_prefs_from_text(self, session_id: str, q: str):
        m = self.get(session_id)
        ql = (q or "").lower()

        # Cheap/price hints
        if any(x in ql for x in ["cheap", "barato", "económico", "economico", "under", "menos de", "$"]):
            m.prefs["price_sensitive"] = True

        # Simple color extraction
        for color in ["black","white","red","blue","green","pink","gray","grey","silver","gold","negro","blanco","rojo","azul","verde","rosado","gris","plata","dorado"]:
            if re.search(rf"\b{re.escape(color)}\b", ql):
                m.prefs["color"] = color
                break

        # Category hints (very rough)
        for cat in ["headphones","boots","shoes","speaker","tablet","kindle","echo","keyboard","mouse","monitor"]:
            if cat in ql:
                m.prefs["category_hint"] = cat
                break

        m.updated_at = time.time()

    def _gc(self):
        now = time.time()
        dead = [k for k,v in self._store.items() if now - v.updated_at > self.ttl]
        for k in dead:
            del self._store[k]
