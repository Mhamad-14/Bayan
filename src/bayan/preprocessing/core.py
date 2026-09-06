"""Lab 1 starter: versioned bilingual preprocessing for Bayan."""

import re
import unicodedata


PREPROC_VERSION = "1.2.0"


def normalize(text: str) -> str:
    """Return deterministic Bayan normalisation while preserving task signal."""

    # Normalize Unicode characters into a consistent representation
    text = unicodedata.normalize("NFKC", text)

    # Remove Arabic tatweel / kashida
    text = text.replace("ـ", "")

    # Reduce any character repeated 3 or more times to exactly 2
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # Collapse spaces, tabs, and newlines into one space
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing whitespace
    return text.strip()


def mask_pii(text: str) -> str:
    """Mask supported phone numbers and Saudi national-ID-shaped values."""

    # Saudi mobile numbers:
    # 05XXXXXXXX
    # +9665XXXXXXXX
    # 9665XXXXXXXX
    phone_pattern = r"(?<!\d)(?:\+966|966|0)5\d{8}(?!\d)"
    text = re.sub(phone_pattern, "<PHONE>", text)

    # Saudi national-ID-shaped numbers:
    # 10 digits beginning with 1 or 2
    national_id_pattern = r"(?<!\d)[12]\d{9}(?!\d)"
    text = re.sub(national_id_pattern, "<NATIONAL_ID>", text)

    return text


def preprocess(text: str) -> str:
    """Apply the shared train/eval/serve preprocessing contract."""

    # Mask PII before normalization so numbers are not accidentally changed
    text = mask_pii(text)
    text = normalize(text)

    return text