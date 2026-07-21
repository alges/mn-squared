"""
Download real coding-sequence (CDS) data from NCBI RefSeq for Experiment D and convert it to
per-gene codon histograms.

This script fetches RefSeq chromosome/genome records via the NCBI E-utilities `fasta_cds_na`
format, which returns the coding sequence of every annotated CDS feature on a given accession. Each
sequence is parsed into codon counts, using the same 64-codon ordering as `cutg_data.py`.

Before running this script, edit `NCBI_PARAMS["email"]` below to your own address, as required by
the NCBI E-utilities usage policy: https://www.ncbi.nlm.nih.gov/books/NBK25497/

Outputs
-------
  experiments/genomics/data/real_genes.npz
    Keys (shape):
      'ecoli_hists'     -- (N_ecoli, 64) int32
      'human_hists'     -- (N_human, 64) int32
      'yeast_hists'     -- (N_yeast, 64) int32
      'bsubtilis_hists' -- (N_bsubtilis, 64) int32
      'dmel_hists'      -- (N_dmel, 64) int32

This file is not committed to the repository; running this script is required before
`exp_d_real_cds.py` can be used. See `experiments/README.md` for details.

Usage
-----
  python -m experiments.genomics.download_real_genes

Expected runtime: 5-15 minutes (depends on network speed and NCBI load).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import requests

from experiments.genomics.cutg_data import CODON_INDEX


NCBI_BASE: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_PARAMS: dict[str, str] = {
    "tool": "mn_squared_genomics_experiment",
    "email": "REPLACE_WITH_YOUR_EMAIL@example.org",  # required by NCBI E-utilities usage policy
}

MIN_CODONS: int = 20   # discard CDS shorter than this (< 60 nt)
MAX_CODONS: int = 5_000  # discard anomalously long pseudo-genes

# RefSeq genome/chromosome accessions for the five organisms in `cutg_data.ORGANISMS`, chosen as
# complete, well-annotated reference sequences so that the CDS FASTA covers a representative set
# of protein-coding genes.
ACCESSIONS: dict[str, list[str]] = {
    "ecoli": ["NC_000913.3"],  # E. coli K-12 MG1655
    "bsubtilis": ["NC_000964.3"],  # B. subtilis 168
    "yeast": [  # S. cerevisiae S288C, all 16 nuclear chromosomes
        "NC_001133.9", "NC_001134.8", "NC_001135.5", "NC_001136.10",
        "NC_001137.3", "NC_001138.5", "NC_001139.9", "NC_001140.6",
        "NC_001141.2", "NC_001142.9", "NC_001143.9", "NC_001144.5",
        "NC_001145.3", "NC_001146.8", "NC_001147.6", "NC_001148.4",
    ],
    "human": ["NC_000022.11"],  # H. sapiens chromosome 22
    "dmel": ["NC_004354.4"],  # D. melanogaster chromosome 2L
}


def _fetch_cds_fasta(accession: str, retries: int = 3) -> str:
    """Fetch the `fasta_cds_na` response for a RefSeq accession, retrying on transient errors."""
    params = {
        **NCBI_PARAMS,
        "db": "nuccore",
        "id": accession,
        "rettype": "fasta_cds_na",
        "retmode": "text",
    }
    for attempt in range(retries):
        try:
            resp = requests.get(f"{NCBI_BASE}/efetch.fcgi", params=params, timeout=300, stream=True)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            wait = 10 * (attempt + 1)
            print(f"    [retry {attempt + 1}] {exc} -- waiting {wait}s")
            time.sleep(wait)
    return ""  # unreachable


def _parse_fasta_seqs(fasta_text: str) -> list[str]:
    """Return a list of nucleotide sequences from a multi-FASTA string."""
    seqs: list[str] = []
    current: list[str] = []
    for line in fasta_text.splitlines():
        if line.startswith(">"):
            if current:
                seqs.append("".join(current))
            current = []
        else:
            current.append(line.strip().upper())
    if current:
        seqs.append("".join(current))
    return seqs


def _count_codons(seq: str) -> np.ndarray | None:
    """
    Count codons in a CDS nucleotide sequence.

    Returns a (64,) int32 array, or None if the sequence is invalid (length not divisible by 3,
    shorter than `MIN_CODONS`, longer than `MAX_CODONS`, or containing no recognised codons).
    """
    seq = seq.replace("N", "").replace("n", "")  # strip ambiguous bases
    n_nt = len(seq)
    if n_nt % 3 != 0:
        return None
    n_codons = n_nt // 3
    if n_codons < MIN_CODONS or n_codons > MAX_CODONS:
        return None

    counts = np.zeros(64, dtype=np.int32)
    recognised = 0
    for i in range(0, n_nt, 3):
        codon = seq[i:i + 3]
        idx = CODON_INDEX.get(codon)
        if idx is not None:
            counts[idx] += 1
            recognised += 1

    if recognised < MIN_CODONS:
        return None
    return counts


def download_organism(name: str, accessions: list[str]) -> np.ndarray:
    """
    Download and parse CDS sequences for all accessions of one organism.

    Returns
    -------
    Integer array of shape (N, 64), one codon histogram per valid CDS.
    """
    all_hists: list[np.ndarray] = []

    for acc in accessions:
        print(f"  Fetching {acc} ... ", end="", flush=True)
        t0 = time.time()
        fasta = _fetch_cds_fasta(acc)
        seqs = _parse_fasta_seqs(fasta)
        print(f"{len(seqs)} sequences ({time.time() - t0:.1f}s)", flush=True)

        time.sleep(0.4)  # stay within NCBI's unauthenticated rate limit of 3 requests/second

        n_ok = 0
        for seq in seqs:
            hist = _count_codons(seq)
            if hist is not None:
                all_hists.append(hist)
                n_ok += 1

        print(f"    -> {n_ok} valid histograms from {acc}")

    if not all_hists:
        raise RuntimeError(f"No valid CDS sequences found for {name}")

    mat = np.stack(all_hists, axis=0)
    print(f"  {name}: {mat.shape[0]} histograms total, median n = {np.median(mat.sum(axis=1)):.0f} codons")
    return mat


if __name__ == "__main__":
    if NCBI_PARAMS["email"] == "REPLACE_WITH_YOUR_EMAIL@example.org":
        raise SystemExit(
            "Please set NCBI_PARAMS['email'] to your own address before running this script, "
            "as required by the NCBI E-utilities usage policy."
        )

    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "real_genes.npz"

    print("=" * 60)
    print("Downloading real gene CDS sequences from NCBI")
    print(f"Output: {out_path}")
    print("=" * 60)

    results: dict[str, np.ndarray] = {}
    t_total = time.time()

    for org_name, acc_list in ACCESSIONS.items():
        print(f"\n[{org_name}]")
        results[f"{org_name}_hists"] = download_organism(org_name, acc_list)

    np.savez_compressed(str(out_path), **results)
    print(f"\nSaved: {out_path}")
    for key, arr in results.items():
        print(f"  {key}: {arr.shape}")

    print(f"\nTotal download time: {(time.time() - t_total) / 60:.1f} min")
