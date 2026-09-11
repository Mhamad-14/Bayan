"""Lab 3 starter: extractive QA post-processing."""

import numpy as np


def best_span(
    start_logits,
    end_logits,
    offsets,
    *,
    null_score,
    null_threshold,
    max_answer_len=30,
    top_k=20,
):
    """Return the best valid extractive span or an honest null answer."""

    start_logits = np.asarray(start_logits)
    end_logits = np.asarray(end_logits)

    # Take only the strongest candidate starts/ends.
    top_starts = np.argsort(start_logits)[-top_k:][::-1]
    top_ends = np.argsort(end_logits)[-top_k:][::-1]

    best = None
    best_score = float("-inf")

    for start_idx in top_starts:
        if start_idx >= len(offsets):
            continue

        start_offset = offsets[start_idx]

        # Special/padding tokens have no usable character span.
        if start_offset is None:
            continue

        for end_idx in top_ends:
            if end_idx >= len(offsets):
                continue

            end_offset = offsets[end_idx]

            if end_offset is None:
                continue

            # Reject inverted spans.
            if end_idx < start_idx:
                continue

            # Reject answers that are too long.
            if end_idx - start_idx + 1 > max_answer_len:
                continue

            char_start = start_offset[0]
            char_end = end_offset[1]

            if char_end <= char_start:
                continue

            score = float(
                start_logits[start_idx]
                + end_logits[end_idx]
            )

            if score > best_score:
                best_score = score
                best = {
                    "answer": (char_start, char_end),
                    "start_index": int(start_idx),
                    "end_index": int(end_idx),
                    "score": score,
                }

    # No valid candidate at all.
    if best is None:
        return {
            "answer": None,
            "score": float(null_score),
        }

    # Honest no-answer path:
    # choose null when its score beats the best span
    # by at least the configured threshold.
    if float(null_score) - best_score >= float(null_threshold):
        return {
            "answer": None,
            "score": float(null_score),
        }

    return best