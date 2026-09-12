"""Lab 7: CPU inference benchmark over the supplied production length mix."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "artifacts/topic_classifier"
SERVING_DIR = ROOT / "artifacts/serving"
BENCH_MIX = ROOT / "data/serving/bench_mix.npy"
BENCHMARKS = ROOT / "BENCHMARKS.md"
QUALITY = ROOT / "artifacts/lab7_quality.json"

THREADS = int(os.environ.get("OMP_NUM_THREADS", "4"))
MAX_CASES = int(os.environ.get("BAYAN_BENCH_CASES", "160"))
WARMUP = int(os.environ.get("BAYAN_BENCH_WARMUP", "10"))

torch.set_num_threads(THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def _load_mix() -> list[str]:
    arr = np.load(BENCH_MIX, allow_pickle=True)
    values = list(np.asarray(arr, dtype=object).reshape(-1))
    texts: list[str] = []

    for item in values:
        if isinstance(item, bytes):
            item = item.decode("utf-8", errors="ignore")

        if isinstance(item, str):
            text = item.strip()
            if text:
                texts.append(text)
            continue

        if isinstance(item, dict):
            for key in ("text", "feedback", "message", "content"):
                if key in item and str(item[key]).strip():
                    texts.append(str(item[key]).strip())
                    break
            continue

        if np.isscalar(item):
            try:
                n = max(4, min(int(float(item)), 512))
            except Exception:
                continue
            words = max(2, n // 5)
            texts.append(("الخدمة تحتاج معالجة عاجلة " * words).strip())

    if not texts:
        raise RuntimeError("Could not derive benchmark texts from data/serving/bench_mix.npy")

    if len(texts) > MAX_CASES:
        idx = np.linspace(0, len(texts) - 1, MAX_CASES, dtype=int)
        texts = [texts[i] for i in idx]

    return texts


def _stats(times_ms: list[float]) -> dict:
    arr = np.asarray(times_ms, dtype=float)
    return {
        "n": int(len(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
    }


def _bench_torch(texts: list[str], *, max_length: int, padded: bool) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    def run(text: str):
        encoded = tokenizer(
            text,
            padding="max_length" if padded else False,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.inference_mode():
            model(**encoded)

    for text in texts[:WARMUP]:
        run(text)

    times = []
    for text in texts:
        start = time.perf_counter_ns()
        run(text)
        end = time.perf_counter_ns()
        times.append((end - start) / 1e6)

    return _stats(times)


def _bench_onnx(texts: list[str], model_path: Path, *, max_length: int = 128) -> dict:
    import onnxruntime as ort

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
        sess_options=_ort_options(),
    )
    input_names = {x.name for x in session.get_inputs()}

    def run(text: str):
        encoded = tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        feed = {
            k: v.astype(np.int64, copy=False)
            for k, v in encoded.items()
            if k in input_names
        }
        session.run(None, feed)

    for text in texts[:WARMUP]:
        run(text)

    times = []
    for text in texts:
        start = time.perf_counter_ns()
        run(text)
        end = time.perf_counter_ns()
        times.append((end - start) / 1e6)

    return _stats(times)


def _ort_options():
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = THREADS
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return opts


def benchmark(*args, **kwargs):
    """Run all currently available Lab 7 optimisation rungs."""
    texts = _load_mix()
    results = {
        "threads": THREADS,
        "source": str(BENCH_MIX.relative_to(ROOT)),
        "n_cases": len(texts),
        "rungs": {},
    }

    print(f"CPU threads: {THREADS}")
    print(f"Benchmark cases from production mix: {len(texts)}")

    print("\n[1/4] torch fp32 @512 padded")
    results["rungs"]["fp32_torch_512_padded"] = _bench_torch(
        texts, max_length=512, padded=True
    )
    print(results["rungs"]["fp32_torch_512_padded"])

    print("\n[2/4] torch fp32 @128 dynamic")
    results["rungs"]["fp32_torch_128_dynamic"] = _bench_torch(
        texts, max_length=128, padded=False
    )
    print(results["rungs"]["fp32_torch_128_dynamic"])

    fp32_onnx = SERVING_DIR / "topic_classifier_fp32.onnx"
    int8_onnx = SERVING_DIR / "topic_classifier_int8.onnx"

    if fp32_onnx.exists():
        print("\n[3/4] ONNX fp32 @128 dynamic")
        results["rungs"]["onnx_fp32_128"] = _bench_onnx(texts, fp32_onnx)
        print(results["rungs"]["onnx_fp32_128"])
    else:
        print("\n[3/4] ONNX fp32 not exported yet — baseline-only run")

    if int8_onnx.exists():
        print("\n[4/4] ONNX INT8 @128 dynamic")
        results["rungs"]["onnx_int8_128"] = _bench_onnx(texts, int8_onnx)
        print(results["rungs"]["onnx_int8_128"])
    else:
        print("\n[4/4] ONNX INT8 not exported yet")

    baseline = results["rungs"]["fp32_torch_512_padded"]["p99_ms"]
    for rung in results["rungs"].values():
        rung["speedup_vs_512_p99"] = (
            baseline / rung["p99_ms"] if rung["p99_ms"] > 0 else None
        )

    SERVING_DIR.mkdir(parents=True, exist_ok=True)
    (SERVING_DIR / "benchmark_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _update_benchmarks(results)
    return results


def _fmt_ms(value):
    return "—" if value is None else f"{value:.2f} ms"


def _fmt_speed(value):
    return "—" if value is None else f"{value:.2f}×"


def _artifact_size(path: Path):
    return "—" if not path.exists() else f"{path.stat().st_size / (1024**2):.1f} MB"


def _update_benchmarks(results: dict):
    quality = {}
    if QUALITY.exists():
        quality = json.loads(QUALITY.read_text(encoding="utf-8"))

    r = results["rungs"]

    def row(key, name, artefact=None):
        d = r.get(key)
        if not d:
            return f"| {name} | — | — | — | — |"
        q = "—"
        if key == "onnx_int8_128" and quality:
            q = (
                f"macro-F1 {quality.get('int8_macro_f1', float('nan')):.4f}; "
                f"tax {quality.get('macro_f1_tax', float('nan')):+.4f}; "
                f"95% CI [{quality.get('tax_ci_low', float('nan')):+.4f}, "
                f"{quality.get('tax_ci_high', float('nan')):+.4f}]"
            )
        return (
            f"| {name} | {_fmt_ms(d['p50_ms'])} | {_fmt_ms(d['p99_ms'])} | "
            f"{q}; speed-up {_fmt_speed(d.get('speedup_vs_512_p99'))} | "
            f"{_artifact_size(artefact) if artefact else '—'} |"
        )

    section = f"""## Lab 7 — Optimisation ladder
CPU evidence: `OMP_NUM_THREADS={THREADS}`, {results['n_cases']} requests sampled deterministically from `{results['source']}`.

| Rung | p50 | p99 | quality metric / paired Δ | Artefact size |
|---|---:|---:|---|---:|
{row('fp32_torch_512_padded', 'fp32 torch @512 padded')}
{row('fp32_torch_128_dynamic', 'fp32 torch @128 dynamic')}
{row('onnx_fp32_128', 'ONNX fp32 @128', SERVING_DIR / 'topic_classifier_fp32.onnx')}
{row('onnx_int8_128', 'ONNX INT8 @128', SERVING_DIR / 'topic_classifier_int8.onnx')}

- classifier bare p99 target <= 25 ms: {"pending" if "onnx_int8_128" not in r else ("met" if r["onnx_int8_128"]["p99_ms"] <= 25 else "not met")}
- speed-up target >= 6x: {"pending" if "onnx_int8_128" not in r else ("met" if r["onnx_int8_128"]["speedup_vs_512_p99"] >= 6 else "not met")}
- classifier quality-tax target <= 0.01 macro-F1: {"pending" if not quality else ("met" if quality.get("macro_f1_tax", 999) <= 0.01 else "not met")}
- HTTP p99, 16 concurrent: pending load test
- classifier quantisation decision: {"pending" if "onnx_int8_128" not in r or not quality else ("use INT8" if quality.get("macro_f1_tax", 999) <= 0.01 else "retain ONNX fp32")}
- NER quantisation decision: see `artifacts/serving/ner_decision.json` if a local Lab 3 NER artefact is available.

"""

    text = BENCHMARKS.read_text(encoding="utf-8") if BENCHMARKS.exists() else ""
    start = text.find("## Lab 7 — Optimisation ladder")
    if start >= 0:
        # Replace Lab 7 through the next top-level Lab heading, or EOF.
        next_pos = text.find("\n## ", start + 5)
        text = text[:start] + section + (text[next_pos + 1:] if next_pos >= 0 else "")
    else:
        text = text.rstrip() + "\n\n" + section

    BENCHMARKS.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    out = benchmark()
    print("\nSaved: artifacts/serving/benchmark_results.json")
