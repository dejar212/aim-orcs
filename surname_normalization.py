"""Normalization pipeline for surnames.

Rules:
- lowercase, `ё->е`, `й->и`
- collapse 3+ identical chars to 1
- transliterate Latin homographs to Cyrillic; other non-Russian -> '*'
- hyphen -> space; double surnames processed as parts
- keep only Cyrillic letters, spaces, '*'
- drop values with zero letters or letters < specials
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from normalization import (
    HOMOGLYPHS,
    NOISE_VALUES,
    RUS_LETTERS,
    SPACE_PATTERN,
    TRIPLE_REPEATS,
    normalize_name,  # reuse homograph rules
)

# local helpers --------------------------------------------------------------

def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _normalize_part(text: str) -> Optional[str]:
    if not text:
        return None
    text = text.lower()
    text = text.replace("ё", "е").replace("й", "и")
    if text in NOISE_VALUES:
        return None

    mapped = []
    for ch in text:
        if ch in HOMOGLYPHS:
            mapped.append(HOMOGLYPHS[ch])
        elif ch in RUS_LETTERS or ch == " ":
            mapped.append(ch)
        else:
            mapped.append("*")
    normalized = "".join(mapped)
    normalized = TRIPLE_REPEATS.sub(r"\1", normalized)
    normalized = SPACE_PATTERN.sub(" ", normalized).strip()

    filtered = []
    for ch in normalized:
        if ch == " " or ch == "*":
            filtered.append(ch)
        elif ch in RUS_LETTERS:
            filtered.append(ch)
    normalized = "".join(filtered)

    letters = sum(1 for ch in normalized if ch in RUS_LETTERS)
    specials = normalized.count("*")
    if letters == 0 or letters < specials:
        return None
    return normalized if normalized else None


def normalize_surname(value: object) -> Optional[str]:
    if _is_missing(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("-", " ")
    parts = [p.strip() for p in text.split() if p.strip()]
    norm_parts = []
    for part in parts:
        norm = _normalize_part(part)
        if norm:
            norm_parts.append(norm)
    if not norm_parts:
        return None
    return " ".join(norm_parts)


def _load_table(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        if nrows is not None:
            try:
                import pyarrow.dataset as ds
            except ModuleNotFoundError:
                return pd.read_parquet(path, engine="pyarrow").head(nrows)
            return ds.dataset(path, format="parquet").head(nrows).to_pandas()
        return pd.read_parquet(path, engine="pyarrow")
    return pd.read_csv(path, nrows=nrows)


def _save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)


def normalize_series(
    series: Iterable[object],
    processes: int = 1,
) -> list[Optional[str]]:
    data = list(series)
    if processes is None or processes < 1:
        processes = 1
    if processes == 1:
        return [normalize_surname(v) for v in tqdm(data, desc="Normalizing surnames", unit="surname")]
    chunk_size = max(2000, len(data) // (processes * 8) or 1)
    tqdm.write(f"Using {processes} processes, chunksize={chunk_size}")
    with mp.Pool(processes=processes) as pool:
        iterator = pool.imap(normalize_surname, data, chunksize=chunk_size)
        return list(tqdm(iterator, total=len(data), desc="Normalizing surnames", unit="surname"))


def main():
    parser = argparse.ArgumentParser(description="Normalize surname column.")
    parser.add_argument("--input", required=True, help="Input file (parquet/csv)")
    parser.add_argument("--column", default="surname", help="Surname column")
    parser.add_argument(
        "--output",
        default="clean_data/clean_surname/normalized.parquet",
        help="Path to write normalized parquet",
    )
    parser.add_argument(
        "--output-column",
        default="surname_normalized",
        help="Name of the normalized column to add",
    )
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--processes", type=int, default=1)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = _load_table(input_path, nrows=args.nrows)
    if args.column not in df.columns:
        raise KeyError(f"Column '{args.column}' not found in {input_path}")

    tqdm.write(f"Loaded {len(df):,} rows from {input_path}. Normalizing '{args.column}'.")
    normalized = normalize_series(df[args.column].tolist(), processes=args.processes)
    df[args.output_column] = normalized
    _save_table(df, output_path)
    tqdm.write(f"Saved normalized data to {output_path.resolve()}")


if __name__ == "__main__":
    main()







