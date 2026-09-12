"""Bayan FastAPI service — Lab 7 classifier serving path."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bayan.serving.canaries import (
    PREPROCESS_VERSION,
    preprocess_text,
    run_startup_canaries,
)

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "artifacts/topic_classifier"
SERVING_DIR = ROOT / "artifacts/serving"
METRICS = MODEL_DIR / "metrics.json"

app = FastAPI(title="Bayan — Bilingual Citizen-Feedback Intelligence Service")


class ClassifyRequest(BaseModel):
    text: str


class TopicPredictor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        metrics = json.loads(METRICS.read_text(encoding="utf-8"))
        self.id2label = {int(v): str(k) for k, v in metrics["label2id"].items()}

        int8 = SERVING_DIR / "topic_classifier_int8.onnx"
        fp32 = SERVING_DIR / "topic_classifier_fp32.onnx"

        self.session = None
        self.model = None

        if int8.exists() or fp32.exists():
            import onnxruntime as ort

            path = int8 if int8.exists() else fp32
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            self.session = ort.InferenceSession(
                str(path),
                providers=["CPUExecutionProvider"],
                sess_options=opts,
            )
            self.input_names = {x.name for x in self.session.get_inputs()}
            self.artefact = path.name
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
            self.model.eval()
            self.artefact = "torch-fp32"

    def predict(self, text: str):
        clean = preprocess_text(text)
        if not clean:
            raise ValueError("text must not be empty")

        if self.session is not None:
            encoded = self.tokenizer(
                clean,
                padding=False,
                truncation=True,
                max_length=128,
                return_tensors="np",
            )
            feed = {
                k: v.astype(np.int64, copy=False)
                for k, v in encoded.items()
                if k in self.input_names
            }
            logits = self.session.run(None, feed)[0][0]
        else:
            encoded = self.tokenizer(
                clean,
                padding=False,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            with torch.inference_mode():
                logits = self.model(**encoded).logits[0].cpu().numpy()

        logits = np.asarray(logits, dtype=float)
        shifted = logits - logits.max()
        probs = np.exp(shifted)
        probs = probs / probs.sum()

        idx = int(probs.argmax())
        return {
            "label": self.id2label[idx],
            "confidence": float(probs[idx]),
            "text": clean,
            "artefact": self.artefact,
            "preprocessing_version": PREPROCESS_VERSION,
        }


CANARIES = run_startup_canaries()
PREDICTOR = TopicPredictor()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "canaries": "green" if CANARIES["green"] else "red",
        "preprocessing_version": PREPROCESS_VERSION,
        "artefact": PREDICTOR.artefact,
    }


@app.post("/v1/classify")
def classify(payload: ClassifyRequest):
    try:
        return PREDICTOR.predict(payload.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/entities")
def entities(payload: dict):
    return {
        "status": "not_implemented_in_lab7",
        "message": "NER serving integration remains part of the final capstone.",
    }


@app.post("/v1/search")
def search(payload: dict):
    return {
        "status": "not_implemented_in_lab7",
        "message": "Search serving integration remains part of the final capstone.",
    }


@app.post("/v1/analyse")
def analyse(payload: dict):
    return {
        "status": "not_implemented_in_lab7",
        "message": "Composite analysis remains part of the final capstone.",
    }
