"""
Codon-usage null distributions for the genomics experiments.

These are genome-wide codon-usage tables (per-thousand values) for five organisms with
well-differentiated biases, taken from the Kazusa Codon Usage Tabulation database (CUTG;
Nakamura, Gojobori and Ikemura, 2000). Each table gives a 64-dimensional probability vector over
codons, ordered as in `CODONS` below, which is used as a fully specified null distribution.

Source: https://www.kazusa.or.jp/codon/
"""
from __future__ import annotations

import numpy as np

from experiments.settings import FloatArray


# Standard genetic-code codon order (64 codons).
CODONS: list[str] = [
    "TTT", "TTC", "TTA", "TTG", "CTT", "CTC", "CTA", "CTG",
    "ATT", "ATC", "ATA", "ATG", "GTT", "GTC", "GTA", "GTG",
    "TCT", "TCC", "TCA", "TCG", "CCT", "CCC", "CCA", "CCG",
    "ACT", "ACC", "ACA", "ACG", "GCT", "GCC", "GCA", "GCG",
    "TAT", "TAC", "TAA", "TAG", "CAT", "CAC", "CAA", "CAG",
    "AAT", "AAC", "AAA", "AAG", "GAT", "GAC", "GAA", "GAG",
    "TGT", "TGC", "TGA", "TGG", "CGT", "CGC", "CGA", "CGG",
    "AGT", "AGC", "AGA", "AGG", "GGT", "GGC", "GGA", "GGG",
]
assert len(CODONS) == 64
CODON_INDEX: dict[str, int] = {codon: idx for idx, codon in enumerate(CODONS)}


# Escherichia coli K-12 (taxid 83333).
_ECOLI_PER_THOUSAND: FloatArray = np.array([
    22.4, 17.3, 13.8, 13.1, 12.2, 10.4, 3.9, 52.6,
    30.3, 25.1, 7.5, 27.3, 19.0, 15.2, 11.4, 26.6,
    15.3, 15.8, 12.3, 8.8, 17.3, 12.4, 16.9, 23.3,
    19.0, 23.4, 13.6, 14.0, 16.4, 25.5, 21.0, 30.1,
    18.7, 12.6, 2.1, 0.2, 13.0, 9.7, 15.8, 28.5,
    22.7, 21.9, 34.2, 31.7, 32.0, 19.4, 39.4, 18.5,
    10.5, 6.4, 0.9, 3.1, 20.7, 21.4, 6.3, 5.9,
    15.3, 16.0, 11.5, 5.7, 24.6, 29.5, 11.1, 15.5,
], dtype=np.float64)

# Homo sapiens (taxid 9606).
_HUMAN_PER_THOUSAND: FloatArray = np.array([
    17.6, 20.3, 7.7, 12.9, 13.2, 19.6, 7.2, 39.6,
    16.0, 20.8, 7.5, 22.1, 11.0, 14.5, 7.1, 28.9,
    15.2, 17.7, 12.2, 4.4, 17.5, 19.8, 16.9, 6.9,
    13.1, 18.9, 15.1, 6.1, 18.4, 27.7, 15.8, 18.1,
    12.1, 15.3, 1.0, 0.6, 10.9, 15.1, 12.3, 34.2,
    17.0, 19.1, 24.4, 31.9, 21.8, 25.1, 29.0, 39.6,
    10.6, 12.6, 1.6, 13.2, 4.5, 10.4, 6.2, 11.4,
    15.2, 19.5, 11.5, 11.4, 10.8, 22.2, 16.5, 16.5,
], dtype=np.float64)

# Saccharomyces cerevisiae (taxid 4932).
_YEAST_PER_THOUSAND: FloatArray = np.array([
    26.1, 18.4, 26.2, 27.2, 12.3, 10.7, 14.2, 11.4,
    30.1, 17.2, 17.8, 21.0, 18.8, 8.4, 11.8, 19.5,
    23.5, 14.2, 23.0, 8.6, 13.5, 6.4, 44.2, 6.4,
    20.0, 22.3, 30.8, 10.6, 21.2, 13.5, 29.0, 12.2,
    18.9, 14.8, 1.3, 0.2, 13.6, 7.8, 30.4, 22.4,
    36.1, 25.0, 42.0, 30.8, 37.6, 21.2, 45.6, 19.2,
    11.6, 6.3, 0.5, 10.4, 6.4, 5.3, 11.4, 5.0,
    14.2, 10.9, 47.5, 21.4, 24.8, 9.5, 25.0, 12.4,
], dtype=np.float64)

# Bacillus subtilis (taxid 1423).
_BSUBTILIS_PER_THOUSAND: FloatArray = np.array([
    27.8, 15.0, 29.5, 22.2, 13.3, 9.6, 8.3, 10.9,
    30.7, 18.2, 16.9, 28.1, 25.0, 10.4, 20.6, 19.2,
    18.0, 11.2, 15.1, 9.2, 14.8, 7.4, 18.2, 10.4,
    20.3, 17.5, 22.6, 8.0, 22.8, 18.4, 33.0, 18.8,
    22.8, 13.2, 2.9, 0.5, 14.5, 7.3, 28.2, 13.6,
    35.7, 25.2, 38.2, 18.1, 37.1, 21.6, 43.9, 20.5,
    10.5, 6.0, 0.9, 9.7, 12.3, 9.0, 4.9, 4.4,
    14.8, 12.0, 22.6, 11.8, 22.8, 16.2, 27.0, 13.4,
], dtype=np.float64)

# Drosophila melanogaster (taxid 7227).
_DMEL_PER_THOUSAND: FloatArray = np.array([
    16.8, 20.5, 5.6, 16.3, 13.6, 19.2, 5.5, 42.3,
    16.4, 23.7, 8.8, 26.6, 10.0, 16.4, 9.2, 27.3,
    12.2, 17.1, 8.2, 7.6, 13.1, 15.4, 14.2, 14.1,
    13.2, 21.0, 12.7, 7.3, 18.0, 25.9, 14.5, 27.0,
    12.4, 17.0, 1.2, 0.3, 10.9, 15.2, 12.4, 34.0,
    17.0, 22.6, 19.6, 29.4, 23.0, 27.5, 27.0, 31.6,
    10.1, 13.6, 0.9, 15.9, 7.2, 17.2, 4.4, 12.8,
    14.4, 21.7, 8.6, 9.5, 11.8, 23.4, 14.0, 20.2,
], dtype=np.float64)


def _normalise(raw: FloatArray) -> FloatArray:
    """Convert per-thousand codon-usage values to a probability vector summing to one."""
    assert raw.shape == (64,), f"expected 64 values, got {raw.shape}"
    return raw / raw.sum()


ORGANISMS: list[dict] = [
    {"name": "E. coli", "probs": _normalise(_ECOLI_PER_THOUSAND)},
    {"name": "H. sapiens", "probs": _normalise(_HUMAN_PER_THOUSAND)},
    {"name": "S. cerevisiae", "probs": _normalise(_YEAST_PER_THOUSAND)},
    {"name": "B. subtilis", "probs": _normalise(_BSUBTILIS_PER_THOUSAND)},
    {"name": "D. melanogaster", "probs": _normalise(_DMEL_PER_THOUSAND)},
]

# Short keys used for filenames and NCBI-derived data lookups (see download_real_genes.py).
ORGANISM_KEYS: list[str] = ["ecoli", "human", "yeast", "bsubtilis", "dmel"]

NULL_PROBABILITIES: FloatArray = np.stack([org["probs"] for org in ORGANISMS], axis=0)  # (5, 64)
