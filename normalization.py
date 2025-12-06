"""Normalization pipeline for Russian first names.

Processes a single name column:
- lowercases, replaces `ё`→`е`, `й`→`и`
- filters noise tokens
- transliterates Latin look‑alike letters to Cyrillic
- replaces any other non‑Russian characters (except space/hyphen) with '*'
- splits hyphenated names into separate parts (keeps spaces)
- drops entries with no letters or with fewer letters than special symbols

The script is importable (exports `normalize_name`) and can be run as CLI:
`python normalization.py --input data/aim-2025-orcs/employees.parquet --column name --output clean_data/clean_name/normalized.parquet`
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

# Basic resources ------------------------------------------------------------

# Tokens that explicitly mean "no data"
NOISE_VALUES = {"none", "нету", "нет", "отсутствует", "ноне", "null", "nan", ""}

# Visually similar Latin -> Cyrillic (covering both cases)
HOMOGLYPHS = {
    "a": "а",
    "A": "а",
    "b": "ь",  # rough visual similarity
    "B": "в",
    "c": "с",
    "C": "с",
    "e": "е",
    "E": "е",
    "k": "к",
    "K": "к",
    "m": "м",
    "M": "м",
    "h": "н",
    "H": "н",
    "o": "о",
    "O": "о",
    "p": "р",
    "P": "р",
    "x": "х",
    "X": "х",
    "y": "у",
    "Y": "у",
    "t": "т",
    "T": "т",
    "i": "и",
    "I": "и",
    "l": "л",
    "L": "л",
}

# Target alphabet after normalization (ё→е, й→и)
RUS_LETTERS = set("абвгдежзиклмнопрстуфхцчшщъыьэюя")

# Translation table for fast character mapping
HOMOGLYPH_TABLE = str.maketrans(HOMOGLYPHS)

SPACE_PATTERN = re.compile(r"\s+")
TRIPLE_REPEATS = re.compile(r"(.)\1{2,}")


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def normalize_name(value: object) -> Optional[str]:
    """Normalize a single name value according to the rules."""
    if _is_missing(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    text = text.lower()
    text = text.replace("ё", "е").replace("й", "и")
    if text in NOISE_VALUES:
        return None

    # Split hyphens into spaces so parts are handled independently
    text = text.replace("-", " ")

    # Transliterate homographs and replace everything non-Cyrillic with '*'
    mapped_chars = []
    for ch in text:
        if ch in HOMOGLYPHS:
            mapped_chars.append(HOMOGLYPHS[ch])
        elif ch in RUS_LETTERS or ch == " ":
            mapped_chars.append(ch)
        else:
            mapped_chars.append("*")

    normalized = "".join(mapped_chars)
    # Collapse 3+ identical symbols to a single one
    normalized = TRIPLE_REPEATS.sub(r"\1", normalized)
    normalized = SPACE_PATTERN.sub(" ", normalized).strip()

    # Keep only valid symbols: Russian letters, space, '*'
    filtered = []
    for ch in normalized:
        if ch == " " or ch == "*":
            filtered.append(ch)
        elif ch in RUS_LETTERS:
            filtered.append(ch)
        # everything else is skipped (already covered)

    normalized = "".join(filtered)

    letters = sum(1 for ch in normalized if ch in RUS_LETTERS)
    specials = normalized.count("*")
    if letters == 0 or letters < specials:
        return None

    return normalized if normalized else None


def _load_table(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        if nrows is not None:
            try:
                import pyarrow.dataset as ds
            except ModuleNotFoundError:
                # Fallback: read fully then head (less efficient but safe)
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
    """Normalize a sequence of values with progress bar."""
    data = list(series)
    if processes is None or processes < 1:
        processes = 1

    if processes == 1:
        return [normalize_name(v) for v in tqdm(data, desc="Normalizing", unit="name")]

    chunk_size = max(2000, len(data) // (processes * 8) or 1)
    tqdm.write(f"Using {processes} processes, chunksize={chunk_size}")
    with mp.Pool(processes=processes) as pool:
        iterator = pool.imap(normalize_name, data, chunksize=chunk_size)
        return list(tqdm(iterator, total=len(data), desc="Normalizing", unit="name"))


def main():
    parser = argparse.ArgumentParser(description="Normalize name column.")
    parser.add_argument("--input", required=True, help="Input file (parquet/csv)")
    parser.add_argument(
        "--column", default="name", help="Column with raw names to normalize"
    )
    parser.add_argument(
        "--output",
        default="clean_data/clean_name/normalized.parquet",
        help="Path to write normalized parquet",
    )
    parser.add_argument(
        "--output-column",
        default="name_normalized",
        help="Name of the normalized column to add",
    )
    parser.add_argument(
        "--nrows", type=int, default=None, help="Optional row limit for testing"
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Number of processes (1 = single process, recommended)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = _load_table(input_path, nrows=args.nrows)
    if args.column not in df.columns:
        raise KeyError(f"Column '{args.column}' not found in {input_path}")

    tqdm.write(
        f"Loaded {len(df):,} rows from {input_path}. Normalizing '{args.column}'."
    )
    normalized = normalize_series(df[args.column].tolist(), processes=args.processes)
    df[args.output_column] = normalized

    _save_table(df, output_path)
    tqdm.write(f"Saved normalized data to {output_path.resolve()}")


if __name__ == "__main__":
    main()

