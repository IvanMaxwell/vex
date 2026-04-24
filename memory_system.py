"""
memory_system.py — Three-layer memory:
  1. ShortTermMemory  – in-process dict (cleared each session)
  2. LongTermMemory   – JSON file (persists user preferences & task history)
  3. VectorMemory     – ChromaDB + all-MiniLM-L2-v2 (similarity search)
"""
import json
import os
import uuid
from datetime import datetime

from config import LT_MEMORY_FILE, CHROMA_DIR


# ══════════════════════════════════════════════════════════════
#  1. SHORT-TERM (working memory)
# ══════════════════════════════════════════════════════════════
class ShortTermMemory:
    def __init__(self):
        self._data: dict         = {}
        self.preferences: dict   = {}
        self.session_results: list[str] = []

    def store(self, key: str, value):
        self._data[key] = value

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def add_preference(self, key: str, value):
        self.preferences[key] = value

    def add_result(self, result: str):
        self.session_results.append(result)
        if len(self.session_results) > 30:
            self.session_results = self.session_results[-30:]

    def get_context(self) -> str:
        parts = []
        if self.preferences:
            parts.append(f"Session preferences: {json.dumps(self.preferences)}")
        if self.session_results:
            parts.append("Recent results: " + " | ".join(self.session_results[-5:]))
        return "\n".join(parts)

    def clear_session(self):
        self._data = {}
        self.session_results = []
        # keep preferences across sessions


# ══════════════════════════════════════════════════════════════
#  2. LONG-TERM (persistent JSON)
# ══════════════════════════════════════════════════════════════
class LongTermMemory:
    def __init__(self):
        self._file = LT_MEMORY_FILE
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self._file):
            try:
                with open(self._file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"user_preferences": {}, "task_history": [], "learned_facts": {}}

    def _save(self):
        with open(self._file, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    def update_preference(self, key: str, value):
        self._data["user_preferences"][key] = value
        self._save()

    def add_task_result(self, task: str, result: str, success: bool):
        self._data["task_history"].append({
            "task":      task[:200],
            "result":    result[:300],
            "success":   success,
            "timestamp": datetime.now().isoformat(),
        })
        self._data["task_history"] = self._data["task_history"][-100:]
        self._save()

    def learn_fact(self, key: str, fact: str):
        self._data["learned_facts"][key] = fact
        self._save()

    def get_preferences(self) -> dict:
        return self._data.get("user_preferences", {})

    def get_context(self) -> str:
        parts = []
        prefs = self._data.get("user_preferences", {})
        if prefs:
            parts.append(f"User preferences: {json.dumps(prefs)}")
        facts = self._data.get("learned_facts", {})
        if facts:
            recent_facts = dict(list(facts.items())[-5:])
            parts.append(f"Known facts: {json.dumps(recent_facts)}")
        history = [t for t in self._data.get("task_history", [])[-10:] if t["success"]]
        if history:
            parts.append("Past successful tasks: " + ", ".join(f'"{t["task"]}"' for t in history[-5:]))
        return "\n".join(parts)


# ══════════════════════════════════════════════════════════════
#  3. VECTOR MEMORY (ChromaDB + MiniLM)
# ══════════════════════════════════════════════════════════════
class VectorMemory:
    def __init__(self):
        self.available = False
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            import chromadb
            from chromadb.utils import embedding_functions
            self._client = chromadb.PersistentClient(path=CHROMA_DIR)
            self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L2-v2"
            )
            self._col = self._client.get_or_create_collection(
                name="agent_memory",
                embedding_function=self._ef,
            )
            self.available = True
            print("[VectorMemory] ChromaDB + MiniLM-L2-v2 ready ✓")
        except Exception as e:
            print(f"[VectorMemory] Not available ({e}). Continuing without it.")

    def store(self, text: str, metadata: dict | None = None):
        if not self.available:
            return
        try:
            self._col.add(
                documents=[text],
                metadatas=[metadata or {}],
                ids=[str(uuid.uuid4())],
            )
        except Exception as e:
            print(f"[VectorMemory] store error: {e}")

    def search(self, query: str, n: int = 3) -> list[str]:
        if not self.available:
            return []
        try:
            count = self._col.count()
            if count == 0:
                return []
            results = self._col.query(
                query_texts=[query],
                n_results=min(n, count),
            )
            return results["documents"][0] if results["documents"] else []
        except Exception as e:
            print(f"[VectorMemory] search error: {e}")
            return []


# ══════════════════════════════════════════════════════════════
#  COMBINED MANAGER
# ══════════════════════════════════════════════════════════════
class MemoryManager:
    def __init__(self):
        self.short_term = ShortTermMemory()
        self.long_term  = LongTermMemory()
        self.vector     = VectorMemory()

    def get_full_context(self, task: str) -> str:
        parts: list[str] = []

        st_ctx = self.short_term.get_context()
        if st_ctx:
            parts.append(f"[Working Memory]\n{st_ctx}")

        lt_ctx = self.long_term.get_context()
        if lt_ctx:
            parts.append(f"[Long-term Memory]\n{lt_ctx}")

        relevant = self.vector.search(task)
        if relevant:
            parts.append("[Relevant Past Experience]\n" + "\n---\n".join(relevant[:2]))

        return "\n\n".join(parts) if parts else "No previous context."

    def save_session(self, task: str, result: str, success: bool):
        self.long_term.add_task_result(task, result, success)
        summary = f"Task: {task[:100]} | Result: {result[:200]} | Success: {success}"
        self.vector.store(summary, {"task": task[:80], "success": str(success)})
        self.short_term.clear_session()

    def update_preferences(self, prefs: dict):
        for k, v in prefs.items():
            self.long_term.update_preference(k, v)
            self.short_term.add_preference(k, v)

    def learn(self, key: str, fact: str):
        self.long_term.learn_fact(key, fact)


memory_manager = MemoryManager()
