"""
retrieval.py
Sentence-embedding kNN baseline.

Embeds all training texts with a sentence transformer, then predicts
by majority vote among k nearest neighbors.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np


class RetrievalBaseline:
    """k-NN retrieval baseline using sentence-transformer embeddings."""

    def __init__(self, k: int = 5, model_name: str = "all-MiniLM-L6-v2"):
        self.k = k
        self.model_name = model_name
        self._model = None
        self.texts: List[str] = []
        self.targets: List[Dict[str, int]] = []
        self.embeddings: Optional[np.ndarray] = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, train_jsonl: str | Path) -> None:
        """Load training data and compute embeddings."""
        with open(train_jsonl) as f:
            records = [json.loads(line) for line in f]

        self.texts = [r["text"] for r in records]
        self.targets = [r["target"] for r in records]
        self.embeddings = self.model.encode(self.texts, show_progress_bar=True,
                                            convert_to_numpy=True)
        self.embeddings = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )

    def predict(self, query: str) -> Dict[str, int]:
        """Predict bin indices for a natural-language query."""
        q_emb = self.model.encode([query], convert_to_numpy=True)
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

        # Cosine similarity
        sims = (self.embeddings @ q_emb.T).squeeze()
        top_k_idx = np.argsort(sims)[-self.k:][::-1]

        # Majority vote per field
        result = {}
        for field in ["id", "x_bin", "y_bin", "angle_bin"]:
            values = [self.targets[i][field] for i in top_k_idx]
            result[field] = max(set(values), key=values.count)

        return result

    def predict_with_examples(self, query: str, n: int = 5) -> tuple:
        """Return prediction + top-n examples (for few-shot prompting)."""
        q_emb = self.model.encode([query], convert_to_numpy=True)
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        sims = (self.embeddings @ q_emb.T).squeeze()
        top_idx = np.argsort(sims)[-n:][::-1]

        examples = [
            {"text": self.texts[i], "target": self.targets[i]}
            for i in top_idx
        ]
        prediction = self.predict(query)
        return prediction, examples

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "embeddings.npy", self.embeddings)
        with open(path / "data.pkl", "wb") as f:
            pickle.dump({"texts": self.texts, "targets": self.targets,
                         "k": self.k, "model_name": self.model_name}, f)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        self.embeddings = np.load(path / "embeddings.npy")
        with open(path / "data.pkl", "rb") as f:
            data = pickle.load(f)
        self.texts = data["texts"]
        self.targets = data["targets"]
        self.k = data["k"]
        self.model_name = data["model_name"]
