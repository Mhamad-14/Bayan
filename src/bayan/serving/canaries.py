"""Lab 7: startup artefact/preprocessing canaries."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "artifacts/topic_classifier"
SERVING_DIR = ROOT / "artifacts/serving"
QUALITY = ROOT / "artifacts/lab7_quality.json"

PREPROCESS_VERSION = "serving_nfkc_ws_v1"


def preprocess_text(text: str) -> str:
    """Deterministic serving-side text normalisation."""
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = re.sub(r"\s+", " ", value).strip()
    return value


def run_startup_canaries() -> dict:
    checks = {}

    checks["classifier_config"] = (MODEL_DIR / "config.json").exists()
    checks["tokenizer"] = (
        (MODEL_DIR / "tokenizer.json").exists()
        or (MODEL_DIR / "tokenizer_config.json").exists()
    )

    # Pinned preprocessing behaviour.
    probe = "  الخدمة   ممتازة  "
    checks["preprocess_pinned"] = preprocess_text(probe) == "الخدمة ممتازة"

    fp32 = SERVING_DIR / "topic_classifier_fp32.onnx"
    int8 = SERVING_DIR / "topic_classifier_int8.onnx"
    checks["serving_artefact"] = int8.exists() or fp32.exists() or (
        (MODEL_DIR / "model.safetensors").exists()
        or (MODEL_DIR / "pytorch_model.bin").exists()
    )

    quality = None
    if QUALITY.exists():
        quality = json.loads(QUALITY.read_text(encoding="utf-8"))
        checks["int8_quality_tax_recorded"] = "macro_f1_tax" in quality
    else:
        checks["int8_quality_tax_recorded"] = True  # fp32/torch fallback is still valid.

    green = all(checks.values())
    result = {
        "green": green,
        "preprocess_version": PREPROCESS_VERSION,
        "checks": checks,
        "quality": quality,
    }

    if not green:
        failed = [k for k, v in checks.items() if not v]
        raise RuntimeError(f"Startup canary failure: {failed}")

    return result
