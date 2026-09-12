"""Lab 5: two-stage bilingual case search."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from bayan.search.index import normalise_search_text


DEFAULT_CROSS_ENCODER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype("float64"), -40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype("float32")


class CaseSearch:
    def __init__(
        self,
        prefix: str = "artifacts/search/case_index_v1",
        *,
        model: str | None = None,
        preproc_version: str | None = None,
        cross_encoder_model: str | None = None,
    ):
        self.prefix = str(prefix)

        manifest_path = Path(f"{prefix}_manifest.json")
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing search manifest: {manifest_path}. Build the index first."
            )

        self.manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

        for key in ["model", "preproc_version", "n_vectors", "dim"]:
            if key not in self.manifest:
                raise ValueError(f"Manifest missing required key: {key}")

        if model is not None and model != self.manifest["model"]:
            raise ValueError("Requested model does not match persisted manifest.")

        if (
            preproc_version is not None
            and preproc_version != self.manifest["preproc_version"]
        ):
            raise ValueError(
                "Requested preprocessing version does not match persisted manifest."
            )

        self.index = faiss.read_index(self.manifest["faiss_path"])

        self.metadata = []
        with open(self.manifest["metadata_path"], encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))

        if len(self.metadata) != int(self.manifest["n_vectors"]):
            raise ValueError("Metadata count does not match FAISS vector count.")

        self.encoder = SentenceTransformer(self.manifest["model"])

        self.cross_encoder_model = (
            cross_encoder_model or DEFAULT_CROSS_ENCODER
        )
        self.cross_encoder = CrossEncoder(self.cross_encoder_model)

    def _encode_query(self, query: str, *, normalize: bool = True) -> np.ndarray:
        clean = normalise_search_text(query)

        vector = self.encoder.encode(
            [clean],
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype("float32")

        if normalize:
            faiss.normalize_L2(vector)

        return vector

    def retrieve(self, query: str, k: int = 10) -> list[dict]:
        vector = self._encode_query(query, normalize=True)

        scores, ids = self.index.search(
            vector,
            min(int(k), int(self.index.ntotal)),
        )

        rows = []
        for score, row_id in zip(scores[0], ids[0]):
            if row_id < 0:
                continue

            item = dict(self.metadata[int(row_id)])
            item["bi_score"] = float(score)
            item["score"] = float((float(score) + 1.0) / 2.0)
            rows.append(item)

        return rows

    def _rerank(self, query: str, items: list[dict]) -> list[dict]:
        if not items:
            return []

        pairs = [(query, item["case_text"]) for item in items]

        raw = np.asarray(
            self.cross_encoder.predict(
                pairs,
                show_progress_bar=False,
            )
        ).reshape(-1)

        cross_conf = _sigmoid(raw)

        reranked = []
        for item, ce_score in zip(items, cross_conf):
            row = dict(item)

            bi_conf = float(
                np.clip(
                    (float(row["bi_score"]) + 1.0) / 2.0,
                    0.0,
                    1.0,
                )
            )

            # Conservative fixed fusion: the cross-encoder reranks while the
            # bi-encoder remains the dominant signal for this synthetic slice.
            final_score = (
                0.80 * bi_conf
                + 0.20 * float(ce_score)
            )

            row["cross_score"] = float(ce_score)
            row["score"] = float(final_score)
            reranked.append(row)

        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked

    def search(
        self,
        query: str,
        k: int = 5,
        candidates: int = 50,
        min_score: float = 0.25,
    ):
        clean_query = normalise_search_text(query)

        candidate_rows = self.retrieve(
            clean_query,
            k=max(int(candidates), int(k)),
        )

        reranked = self._rerank(clean_query, candidate_rows)

        accepted = [
            row
            for row in reranked
            if float(row["score"]) >= float(min_score)
        ]

        return accepted[: int(k)]
