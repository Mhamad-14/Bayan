"""Lab 1 starter: sentence segmentation."""

import spacy

from .core import preprocess


def build_pipeline():
    """Build a lightweight spaCy pipeline for sentence segmentation."""

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

    return nlp


def split_sentences(raw: str, nlp) -> list[str]:
    """Preprocess text and split it into non-empty sentences."""

    cleaned = preprocess(raw)
    doc = nlp(cleaned)

    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
