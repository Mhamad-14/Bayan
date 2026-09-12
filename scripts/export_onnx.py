"""Lab 7: export classifier to ONNX, quantise INT8, and measure paired quality tax."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "artifacts/topic_classifier"
SERVING_DIR = ROOT / "artifacts/serving"
PRED = ROOT / "data/eval/validation_predictions.csv"
RAW = ROOT / "data/raw/bayan_feedback.csv"
METRICS = MODEL_DIR / "metrics.json"

THREADS = int(os.environ.get("OMP_NUM_THREADS", "4"))
QUALITY_N = int(os.environ.get("BAYAN_QUALITY_CASES", "400"))

torch.set_num_threads(THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


class ClassifierWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits


class TokenClassifierWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits


def _export_classifier():
    SERVING_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    rollback = SERVING_DIR / "topic_classifier_fp32_rollback"
    rollback.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(rollback)
    tokenizer.save_pretrained(rollback)

    dummy = tokenizer(
        "الخدمة ممتازة ولكن التأخير طويل",
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    wrapper = ClassifierWrapper(model)
    fp32 = SERVING_DIR / "topic_classifier_fp32.onnx"

    torch.onnx.export(
        wrapper,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(fp32),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
    )

    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8 = SERVING_DIR / "topic_classifier_int8.onnx"
    quantize_dynamic(
        model_input=str(fp32),
        model_output=str(int8),
        weight_type=QuantType.QInt8,
    )

    return fp32, int8


def _label_maps():
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    label2id = {str(k): int(v) for k, v in metrics["label2id"].items()}
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label


def _torch_predictions(texts: list[str]) -> list[str]:
    _, id2label = _label_maps()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    out = []
    for start in range(0, len(texts), 32):
        batch = texts[start:start + 32]
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        with torch.inference_mode():
            pred = model(**enc).logits.argmax(dim=-1).tolist()
        out.extend(id2label[int(i)] for i in pred)
    return out


def _onnx_predictions(texts: list[str], path: Path) -> list[str]:
    import onnxruntime as ort

    _, id2label = _label_maps()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = THREADS
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
        sess_options=opts,
    )
    names = {x.name for x in session.get_inputs()}

    out = []
    for start in range(0, len(texts), 32):
        batch = texts[start:start + 32]
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="np",
        )
        feed = {
            k: v.astype(np.int64, copy=False)
            for k, v in enc.items()
            if k in names
        }
        logits = session.run(None, feed)[0]
        pred = logits.argmax(axis=-1).tolist()
        out.extend(id2label[int(i)] for i in pred)
    return out


def _quality_check(int8_path: Path):
    pred = pd.read_csv(PRED)
    raw = pd.read_csv(RAW, usecols=["feedback_id", "text"])
    df = pred.merge(raw, on="feedback_id", how="left")
    df = df.dropna(subset=["text"]).copy()

    n = min(QUALITY_N, len(df))
    df = df.sample(n=n, random_state=42).reset_index(drop=True)

    texts = df["text"].astype(str).tolist()
    gold = df["y_true"].astype(str).tolist()

    torch_pred = _torch_predictions(texts)
    int8_pred = _onnx_predictions(texts, int8_path)

    fp32_f1 = float(f1_score(gold, torch_pred, average="macro", zero_division=0))
    int8_f1 = float(f1_score(gold, int8_pred, average="macro", zero_division=0))
    tax = fp32_f1 - int8_f1

    rng = np.random.default_rng(42)
    boot_tax = []
    idx = np.arange(len(gold))
    gold_arr = np.asarray(gold, dtype=object)
    torch_arr = np.asarray(torch_pred, dtype=object)
    int8_arr = np.asarray(int8_pred, dtype=object)

    for _ in range(500):
        sample = rng.choice(idx, size=len(idx), replace=True)
        f = f1_score(
            gold_arr[sample],
            torch_arr[sample],
            average="macro",
            zero_division=0,
        )
        q = f1_score(
            gold_arr[sample],
            int8_arr[sample],
            average="macro",
            zero_division=0,
        )
        boot_tax.append(float(f - q))

    lo, hi = np.quantile(boot_tax, [0.025, 0.975])

    result = {
        "n": n,
        "fp32_macro_f1": fp32_f1,
        "int8_macro_f1": int8_f1,
        "macro_f1_tax": tax,
        "tax_ci_low": float(lo),
        "tax_ci_high": float(hi),
        "prediction_agreement": float(np.mean(torch_arr == int8_arr)),
    }

    (ROOT / "artifacts/lab7_quality.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def _find_ner_dir():
    candidates = [
        ROOT / "artifacts/ner",
        ROOT / "artifacts/ner_model",
        ROOT / "artifacts/named_entity_recognition",
    ]
    env = os.environ.get("BAYAN_NER_DIR")
    if env:
        candidates.insert(0, Path(env).expanduser())

    for p in candidates:
        if (p / "config.json").exists() and (
            (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()
        ):
            return p
    return None


def _export_ner_if_available():
    decision_path = SERVING_DIR / "ner_decision.json"
    ner_dir = _find_ner_dir()

    if ner_dir is None:
        decision = {
            "available": False,
            "decision": "not benchmarked locally",
            "reason": (
                "No local Lab 3 NER artefact was found. "
                "Set BAYAN_NER_DIR to the saved Lab 3 NER directory and rerun."
            ),
        }
        decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
        print("NER: local artefact not found; decision recorded without fabricated metrics.")
        return

    tok = AutoTokenizer.from_pretrained(ner_dir)
    model = AutoModelForTokenClassification.from_pretrained(ner_dir)
    model.eval()

    dummy = tok(
        "بلاغ في حي الياسمين بالرياض",
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    wrapper = TokenClassifierWrapper(model)
    fp32 = SERVING_DIR / "ner_fp32.onnx"

    torch.onnx.export(
        wrapper,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(fp32),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
        dynamo=False,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
    )

    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8 = SERVING_DIR / "ner_int8.onnx"
    quantize_dynamic(
        model_input=str(fp32),
        model_output=str(int8),
        weight_type=QuantType.QInt8,
    )

    decision = {
        "available": True,
        "fp32_onnx": str(fp32.relative_to(ROOT)),
        "int8_onnx": str(int8.relative_to(ROOT)),
        "decision": (
            "Both NER artefacts exported. Retain fp32 as rollback; "
            "choose INT8 only after paired NER quality evidence is available."
        ),
    }
    decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(f"NER exported from: {ner_dir}")


def main():
    print("Exporting classifier fp32 ONNX + rollback artefact...")
    fp32, int8 = _export_classifier()
    print(f"fp32: {fp32}")
    print(f"INT8: {int8}")

    print("\nRunning paired classifier quality check...")
    quality = _quality_check(int8)
    print(json.dumps(quality, indent=2))

    print("\nHandling NER...")
    _export_ner_if_available()

    print("\nLab 7 ONNX export complete ✅")


if __name__ == "__main__":
    main()
