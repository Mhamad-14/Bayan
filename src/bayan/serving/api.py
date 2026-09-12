"""Bayan FastAPI service — Lab 7 classifier serving path."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bayan.search.service import CaseSearch

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


class BatchClassifyRequest(BaseModel):
    texts: list[str]


class SearchRequest(BaseModel):
    query: str
    k: int = 5


class EntitiesRequest(BaseModel):
    text: str


class AnalyseRequest(BaseModel):
    text: str
    k: int = 3


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
            opts.intra_op_num_threads = int(os.environ.get("BAYAN_ORT_THREADS", "1"))
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
CLASSIFY_MAX_INFLIGHT = int(os.environ.get("BAYAN_CLASSIFY_MAX_INFLIGHT", "8"))
CLASSIFY_SEMAPHORE = threading.BoundedSemaphore(CLASSIFY_MAX_INFLIGHT)
SEARCHER = None
NER_PIPELINE = None


def get_ner_pipeline():
    global NER_PIPELINE
    if NER_PIPELINE is None:
        ner_dir = ROOT / "artifacts/ner"
        if not ner_dir.exists():
            raise FileNotFoundError("Missing NER artefact. Run: python scripts/train_ner.py")
        from transformers import pipeline
        NER_PIPELINE = pipeline(
            "token-classification",
            model=str(ner_dir),
            tokenizer=str(ner_dir),
            aggregation_strategy="simple",
            device=-1,
        )
    return NER_PIPELINE


def get_searcher():
    global SEARCHER
    if SEARCHER is None:
        SEARCHER = CaseSearch(prefix="artifacts/search/case_index_v1")
    return SEARCHER


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
        with CLASSIFY_SEMAPHORE:
            return PREDICTOR.predict(payload.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/classify:batch")
def classify_batch(payload: BatchClassifyRequest):
    if not payload.texts:
        raise HTTPException(status_code=400, detail="texts must not be empty")
    try:
        return {"results": [PREDICTOR.predict(text) for text in payload.texts]}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/entities")
def entities(payload: EntitiesRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    try:
        raw = get_ner_pipeline()(payload.text)
        results = [
            {
                "entity": str(item["entity_group"]),
                "text": str(item["word"]),
                "score": float(item["score"]),
                "start": int(item["start"]),
                "end": int(item["end"]),
            }
            for item in raw
        ]
        return {"text": payload.text, "entities": results}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/v1/search")
def search(payload: SearchRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    try:
        results = get_searcher().search(
            payload.query,
            k=payload.k,
            candidates=50,
            min_score=0.6651,
        )
        return {"query": payload.query, "results": results}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"search unavailable: {exc}") from exc


@app.post("/v1/analyse")
def analyse(payload: AnalyseRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    try:
        classification = PREDICTOR.predict(payload.text)

        raw_entities = get_ner_pipeline()(payload.text)
        entity_results = [
            {
                "entity": str(item["entity_group"]),
                "text": str(item["word"]),
                "score": float(item["score"]),
                "start": int(item["start"]),
                "end": int(item["end"]),
            }
            for item in raw_entities
        ]

        similar_cases = get_searcher().search(
            payload.text,
            k=payload.k,
            candidates=50,
            min_score=0.6651,
        )

        return {
            "text": payload.text,
            "classification": classification,
            "entities": entity_results,
            "similar_cases": similar_cases,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"analysis unavailable: {exc}") from exc
