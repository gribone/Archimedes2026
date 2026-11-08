"""
Greek Text Normalization & Transliteration (`src/features/text.py`)
===================================================================
Deterministic Greek character normalization (NFD, tonos removal, uppercase,
final sigma normalization) and Greek-to-Latin transliteration tables.
"""

from __future__ import annotations

import unicodedata


def normalize_greek_text(text: str) -> str:
    """
    Deterministic Greek text normalization:
    strips diacritics/tonos (NFD), uppercases, and normalizes final sigma (ς -> Σ).
    """
    nfd = unicodedata.normalize("NFD", text or "")
    stripped = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    return stripped.upper().strip()


GREEK_TO_LATIN_TABLE: dict[str, str] = {
    "Α": "A", "Β": "V", "Γ": "G", "Δ": "D", "Ε": "E", "Ζ": "Z",
    "Η": "I", "Θ": "TH", "Ι": "I", "Κ": "K", "Λ": "L", "Μ": "M",
    "Ν": "N", "Ξ": "X", "Ο": "O", "Π": "P", "Ρ": "R", "Σ": "S",
    "Τ": "T", "Υ": "Y", "Φ": "F", "Χ": "CH", "Ψ": "PS", "Ω": "O",
}


def transliterate_greek(text: str) -> str:
    """Transliterate Greek text into Latin uppercase characters."""
    res = []
    for ch in text.upper():
        res.append(GREEK_TO_LATIN_TABLE.get(ch, ch))
    return "".join(res)


__all__ = [
    "GREEK_TO_LATIN_TABLE",
    "normalize_greek_text",
    "transliterate_greek",
]
