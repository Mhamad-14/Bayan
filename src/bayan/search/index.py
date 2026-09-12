
"""Lab 5: versioned FAISS index over Bayan historical cases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import unicodedata

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DEFAULT_DATA_PATH = "data/search/bayan_cases.csv"
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PREPROC_VERSION = "bayan_search_v1"

_WS_RE = re.compile(r"\s+")


def normalise_search_text(text: str) -> str:
    """Apply minimal multilingual-safe normalization consistently."""
    value = unicodedata.normalize("NFKC", str(text or ""))
    return _WS_RE.sub(" ", value).strip()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_index(
    prefix: str = "artifacts/search/case_index_v1",
    limit: int | None = None,
    *,
    data_path: str = DEFAULT_DATA_PATH,
    model: str = DEFAULT_MODEL,
    preproc_version: str = PREPROC_VERSION,
):
    prefix_path = Path(prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)

    source = Path(data_path)
    df = pd.read_csv(source)

    if limit is not None:
        df = df.head(int(limit)).copy()

    if "case_id" not in df.columns or "case_text" not in df.columns:
        raise ValueError("Case corpus must contain case_id and case_text.")

    texts = [
        normalise_search_text(x)
        for x in df["case_text"].astype(str).tolist()
    ]

    encoder = SentenceTransformer(model)

    raw_vectors = encoder.encode(
        texts,
        batch_size=128,
        show_progress_bar=len(texts) > 100,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype("float32")

    if raw_vectors.ndim != 2 or len(raw_vectors) == 0:
        raise RuntimeError("Encoder returned an invalid embedding matrix.")

    vectors = raw_vectors.copy()
    faiss.normalize_L2(vectors)

    dim = int(vectors.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss_path = Path(f"{prefix}.faiss")
    metadata_path = Path(f"{prefix}.metadata.jsonl")
    manifest_path = Path(f"{prefix}_manifest.json")
    raw_path = Path(f"{prefix}.raw.npy")

    faiss.write_index(index, str(faiss_path))
    np.save(raw_path, raw_vectors)

    with metadata_path.open("w", encoding="utf-8") as f:
        for row_id, (_, row) in enumerate(df.reset_index(drop=True).iterrows()):
            item = {
                "row_id": row_id,
                "case_id": str(row["case_id"]),
                "lang": str(row.get("lang", "")),
                "topic": str(row.get("topic", "")),
                "case_text": str(row.get("case_text", "")),
                "resolution": str(row.get("resolution", "")),
                "status": str(row.get("status", "")),
                "closed_at": str(row.get("closed_at", "")),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    manifest = {
        "version": 1,
        "model": model,
        "preproc_version": preproc_version,
        "n_vectors": int(index.ntotal),
        "dim": dim,
        "metric": "cosine_via_l2_normalized_inner_product",
        "index_type": "IndexFlatIP",
        "corpus_path": str(source),
        "corpus_sha256": _sha256(source),
        "faiss_path": str(faiss_path),
        "metadata_path": str(metadata_path),
        "raw_vectors_path": str(raw_path),
    }

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Built FAISS index")
    print("Vectors:", manifest["n_vectors"])
    print("Dimension:", manifest["dim"])
    print("Model:", manifest["model"])
    print("Preprocessing:", manifest["preproc_version"])
    print("Manifest:", manifest_path)

    return manifest
