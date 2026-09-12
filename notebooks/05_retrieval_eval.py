
"""Lab 5: labelled-query retrieval evaluation."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import faiss
import numpy as np

from bayan.search.service import CaseSearch


PREFIX = "artifacts/search/case_index_v1"
QUERY_PATH = Path("data/search/bayan_queries.jsonl")
BM25_PATH = Path("data/search/bm25_baseline_results.jsonl")
OUT = Path("artifacts/lab5_retrieval_metrics.json")


def read_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def hit_and_rr(retrieved_ids, relevant_ids):
    relevant = set(relevant_ids)
    if not relevant:
        return None, None

    hit = 1.0 if any(x in relevant for x in retrieved_ids[:10]) else 0.0

    rr = 0.0
    for rank, case_id in enumerate(retrieved_ids[:10], start=1):
        if case_id in relevant:
            rr = 1.0 / rank
            break

    return hit, rr


def mean_or_none(values):
    values = [x for x in values if x is not None]
    return float(np.mean(values)) if values else None


def p50_ms(values):
    return float(statistics.median(values)) if values else None


def main():
    queries = read_jsonl(QUERY_PATH)
    searcher = CaseSearch(prefix=PREFIX)

    metadata_by_id = {
        item["case_id"]: item
        for item in searcher.metadata
    }

    answerable = [
        q for q in queries
        if q.get("relevant_case_ids")
    ]
    no_answer = [
        q for q in queries
        if not q.get("relevant_case_ids")
    ]

    print("Queries:", len(queries))
    print("Answerable:", len(answerable))
    print("No-answer:", len(no_answer))
    print("Bi-encoder:", searcher.manifest["model"])
    print("Cross-encoder:", searcher.cross_encoder_model)

    bi_hits, bi_rrs, bi_latency = [], [], []
    rerank_hits, rerank_rrs, rerank_latency = [], [], []
    reranked_cache = {}

    same_lang_hits = []
    cross_lang_hits = []

    for q in queries:
        query = q["query"]

        t0 = time.perf_counter()
        bi = searcher.retrieve(query, k=10)
        bi_latency.append((time.perf_counter() - t0) * 1000.0)

        if q.get("relevant_case_ids"):
            h, rr = hit_and_rr(
                [x["case_id"] for x in bi],
                q["relevant_case_ids"],
            )
            bi_hits.append(h)
            bi_rrs.append(rr)

        t0 = time.perf_counter()
        rr_rows = searcher.search(
            query,
            k=10,
            candidates=50,
            min_score=0.0,
        )
        rerank_latency.append((time.perf_counter() - t0) * 1000.0)
        reranked_cache[q["query_id"]] = rr_rows

        if q.get("relevant_case_ids"):
            ids = [x["case_id"] for x in rr_rows]
            h, rr = hit_and_rr(ids, q["relevant_case_ids"])
            rerank_hits.append(h)
            rerank_rrs.append(rr)

            q_lang = str(q.get("lang", ""))
            same_rel = [
                cid for cid in q["relevant_case_ids"]
                if cid in metadata_by_id
                and metadata_by_id[cid]["lang"] == q_lang
            ]
            cross_rel = [
                cid for cid in q["relevant_case_ids"]
                if cid in metadata_by_id
                and metadata_by_id[cid]["lang"] != q_lang
            ]

            if same_rel:
                same_lang_hits.append(
                    1.0 if set(ids[:10]) & set(same_rel) else 0.0
                )
            if cross_rel:
                cross_lang_hits.append(
                    1.0 if set(ids[:10]) & set(cross_rel) else 0.0
                )

    scored = []
    for q in queries:
        rows = reranked_cache[q["query_id"]]
        top_score = float(rows[0]["score"]) if rows else 0.0
        scored.append(
            {
                "answerable": bool(q.get("relevant_case_ids")),
                "top_score": top_score,
            }
        )

    unique_scores = sorted({x["top_score"] for x in scored})
    threshold_candidates = [0.0, 1.0] + unique_scores
    threshold_candidates += [
        (a + b) / 2.0
        for a, b in zip(unique_scores, unique_scores[1:])
    ]

    best = None
    for threshold in sorted(set(threshold_candidates)):
        na_rows = [x for x in scored if not x["answerable"]]
        a_rows = [x for x in scored if x["answerable"]]

        na_correct = sum(x["top_score"] < threshold for x in na_rows)
        answerable_kept = sum(x["top_score"] >= threshold for x in a_rows)

        na_acc = na_correct / len(na_rows) if na_rows else 1.0
        retention = answerable_kept / len(a_rows) if a_rows else 1.0
        balanced = 0.5 * (na_acc + retention)

        candidate = (
            balanced,
            na_correct,
            answerable_kept,
            -threshold,
            threshold,
            retention,
        )

        if best is None or candidate > best:
            best = candidate

    threshold = float(best[4])
    no_answer_correct = int(best[1])
    answerable_retention = float(best[5])

    same_recall = mean_or_none(same_lang_hits)
    cross_recall = mean_or_none(cross_lang_hits)
    cross_gap = (
        None
        if same_recall is None or cross_recall is None
        else float(same_recall - cross_recall)
    )

    raw_vectors = np.load(
        searcher.manifest["raw_vectors_path"]
    ).astype("float32")

    buggy = faiss.IndexFlatIP(raw_vectors.shape[1])
    buggy.add(raw_vectors)

    bug_hits, bug_rrs = [], []

    for q in answerable:
        raw_q = searcher._encode_query(q["query"], normalize=False)
        _, ids = buggy.search(raw_q, 10)

        retrieved = [
            searcher.metadata[int(i)]["case_id"]
            for i in ids[0]
            if int(i) >= 0
        ]

        h, rr = hit_and_rr(retrieved, q["relevant_case_ids"])
        bug_hits.append(h)
        bug_rrs.append(rr)

    bm25_rows = read_jsonl(BM25_PATH)
    bm25_recall = mean_or_none(
        [float(x.get("recall_at_10", 0.0)) for x in bm25_rows]
    )
    bm25_mrr = mean_or_none(
        [float(x.get("reciprocal_rank", 0.0)) for x in bm25_rows]
    )

    metrics = {
        "n_queries": len(queries),
        "n_answerable": len(answerable),
        "n_no_answer": len(no_answer),
        "bi_encoder": searcher.manifest["model"],
        "cross_encoder": searcher.cross_encoder_model,
        "bm25_baseline": {
            "recall_at_10": bm25_recall,
            "mrr_at_10": bm25_mrr,
        },
        "bi_encoder_only": {
            "recall_at_10": mean_or_none(bi_hits),
            "mrr_at_10": mean_or_none(bi_rrs),
            "p50_ms_per_query": p50_ms(bi_latency),
        },
        "with_cross_encoder_rerank": {
            "recall_at_10": mean_or_none(rerank_hits),
            "mrr_at_10": mean_or_none(rerank_rrs),
            "p50_ms_per_query": p50_ms(rerank_latency),
        },
        "language_slices": {
            "same_language_recall_at_10": same_recall,
            "cross_language_recall_at_10": cross_recall,
            "gap_same_minus_cross": cross_gap,
        },
        "no_answer": {
            "threshold": threshold,
            "correct": no_answer_correct,
            "total": len(no_answer),
            "answerable_retention": answerable_retention,
        },
        "unnormalised_vector_bug": {
            "recall_at_10": mean_or_none(bug_hits),
            "mrr_at_10": mean_or_none(bug_rrs),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== Lab 5 Retrieval Evaluation ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    print("\n=== Benchmark Targets ===")
    print(
        "recall@10 >= 0.80:",
        metrics["with_cross_encoder_rerank"]["recall_at_10"],
    )
    print(
        "MRR@10 >= 0.70:",
        metrics["with_cross_encoder_rerank"]["mrr_at_10"],
    )
    print(
        "no-answer >= 17/20:",
        f'{no_answer_correct}/{len(no_answer)}',
    )

    print("\nSaved:", OUT)


if __name__ == "__main__":
    main()
