"""Lab 4: model-aware Arabic normalization and clitic segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
import unicodedata


@dataclass(frozen=True)
class ArabicProfile:
    name: str
    dediacritize: bool = False


_AR_DIACRITICS_RE = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)
_TATWEEL_RE = re.compile("\u0640+")
_WS_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_USER_RE = re.compile(r"(?<!\w)@\w+")
_CONTROL_RE = re.compile(
    r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]"
)


def _common(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = _CONTROL_RE.sub("", text)
    text = _TATWEEL_RE.sub("", text)
    return _WS_RE.sub(" ", text).strip()


def _dediac(text: str) -> str:
    return _AR_DIACRITICS_RE.sub("", text)


def normalize_arabic(text: str, profile: ArabicProfile) -> str:
    """Normalize Arabic text according to the requested course profile."""
    if text is None:
        return ""

    out = _common(text)
    name = profile.name.strip().lower().replace("_", "-")

    if profile.dediacritize:
        out = _dediac(out)

    # Course Lab 4 normalization profile.
    if name == "bayan-ar-v1":
        translation = str.maketrans({
            "أ": "ا",
            "إ": "ا",
            "آ": "ا",
            "ٱ": "ا",
            "ؤ": "و",
            "ئ": "ي",
            "ى": "ي",
            "ة": "ه",
        })
        out = out.translate(translation)
        return _WS_RE.sub(" ", out).strip()

    # CAMeLBERT-style light normalization.
    if "camel" in name:
        return _WS_RE.sub(" ", out).strip()

    # MARBERT / ARBERT social-media normalization.
    if "marbert" in name or "arbert" in name:
        out = _URL_RE.sub("URL", out)
        out = _USER_RE.sub("USER", out)
        out = re.sub(
            r"#([^\s#]+)",
            lambda m: m.group(1).replace("_", " "),
            out,
        )
        return _WS_RE.sub(" ", out).strip()

    return _WS_RE.sub(" ", out).strip()


def _fallback_segment_word(word: str) -> list[str]:
    if not word:
        return []

    pieces = []
    rest = word

    if len(rest) > 2 and rest.startswith("و"):
        pieces.append("و+")
        rest = rest[1:]

    if len(rest) > 2 and rest.startswith("ف"):
        pieces.append("ف+")
        rest = rest[1:]

    if len(rest) > 2 and rest.startswith("ب"):
        pieces.append("ب+")
        rest = rest[1:]

    if len(rest) > 2 and rest.startswith("ك"):
        pieces.append("ك+")
        rest = rest[1:]

    if len(rest) > 2 and rest.startswith("ل") and not rest.startswith("ال"):
        pieces.append("ل+")
        rest = rest[1:]

    if len(rest) > 2 and rest.startswith("ال"):
        pieces.append("ال+")
        rest = rest[2:]

    if rest:
        pieces.append(rest)

    return pieces


@lru_cache(maxsize=1)
def _camel_tokenizer():
    from camel_tools.disambig.mle import MLEDisambiguator
    from camel_tools.tokenizers.morphological import MorphologicalTokenizer

    mle = MLEDisambiguator.pretrained("calima-msa-r13")

    return MorphologicalTokenizer(
        disambiguator=mle,
        scheme="d3tok",
        split=True,
        diac=False,
    )


def _flatten_tokenized(item) -> list[str]:
    if isinstance(item, str):
        chunks = item.split("_")
    else:
        chunks = []
        for x in item:
            chunks.extend(str(x).split("_"))

    return [x for x in chunks if x]


def segment(text: str) -> list[str]:
    """Segment Arabic clitics with CAMeL Tools D3 tokenization."""
    from camel_tools.tokenizers.word import simple_word_tokenize

    words = [
        w
        for w in simple_word_tokenize(str(text))
        if re.search(r"[\u0600-\u06FF]", w)
    ]

    if not words:
        return []

    try:
        tokenized = _camel_tokenizer().tokenize(words)

        pieces = []

        for item in tokenized:
            pieces.extend(_flatten_tokenized(item))

        return pieces

    except Exception:
        pieces = []

        for word in words:
            pieces.extend(_fallback_segment_word(word))

        return pieces
