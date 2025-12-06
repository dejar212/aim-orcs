"""Fuzzy correction for normalized Russian first names.

Algorithm:
- Build a reference dictionary of normalized names (download + local fallback)
- Precompute all variants within Damerau-Levenshtein distance 1 for the reference
- For each normalized value:
  * empty -> class 'empty'
  * contains '*' -> try single-regex resolution, otherwise ambiguous/no_match
  * exact dictionary hit -> 'untouched'
  * single candidate within distance 1 -> 'corrected'
  * multiple candidates -> 'ambiguous'
  * none -> 'no_match'

Outputs corrected parquet and JSON reports with class distribution and samples.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from tqdm import tqdm

from normalization import RUS_LETTERS, normalize_name

# -----------------------------------------------------------------------------
# Reference data helpers

REFERENCE_URLS = [
    "https://raw.githubusercontent.com/danakt/russian-names/master/russian_names.json",
    # Combined list (both genders)
    "https://raw.githubusercontent.com/Harrix/Russian-Nouns/master/dist/all_names_rus.txt",
]
# Explicit female / male lists to guarantee coverage
FEMALE_URLS = [
    "https://raw.githubusercontent.com/Harrix/Russian-Nouns/master/dist/female_first_names_rus.txt",
]
MALE_URLS = [
    "https://raw.githubusercontent.com/Harrix/Russian-Nouns/master/dist/male_first_names_rus.txt",
]

DEFAULT_DICT_PATH = Path("data/reference/russian_names.txt")
PATRONYMICS_PATH = Path("patronymics.csv")

# Minimal guaranteed female names to avoid drops
MANUAL_FEMALE_NAMES = {
    "юлия",
    "яна",
    "анна",
    "мария",
    "елена",
    "ольга",
    "наталья",
    "екатерина",
    "светлана",
    "дина",
    "таиса",
    "нина",
    "инна",
}


def download_text(url: str, timeout: int = 15) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def parse_names_from_text(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw:
        return []
    # Try JSON first
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            names: list[str] = []
            for item in data:
                if isinstance(item, dict):
                    if "name" in item:
                        names.append(str(item["name"]))
                    # danakt entries may have "sex" but we collect both anyway
                elif isinstance(item, str):
                    names.append(item)
            if names:
                return names
        elif isinstance(data, dict) and "name" in data:
            return [str(data["name"])]
    except json.JSONDecodeError:
        pass

    # Fallback: treat as plain text / CSV-like
    names: list[str] = []
    for line in raw.splitlines():
        for token in re.split(r"[;,]", line):
            token = token.strip()
            if token:
                names.append(token)
    return names


def load_patronymic_names(path: Path = PATRONYMICS_PATH) -> list[str]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
        return df[df.columns[0]].astype(str).tolist()
    except Exception:
        return []


def ensure_reference_names(
    path: Path = DEFAULT_DICT_PATH,
    extra_paths: Optional[list[Path]] = None,
    extra_urls: Optional[list[str]] = None,
) -> set[str]:
    existing: set[str] = set()
    if path.exists():
        existing = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    names: set[str] = set(existing)

    def _extend_from_urls(urls):
        for url in urls:
            try:
                tqdm.write(f"Downloading reference names from {url}")
                raw = download_text(url)
                names.update(parse_names_from_text(raw))
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                tqdm.write(f"Failed to download {url}: {exc}")

    # Always try to enrich (even if file exists) to ensure female names present
    _extend_from_urls(REFERENCE_URLS)
    _extend_from_urls(FEMALE_URLS)
    _extend_from_urls(MALE_URLS)
    if extra_urls:
        _extend_from_urls(extra_urls)

    if extra_paths:
        for p in extra_paths:
            if p.exists():
                try:
                    names.update(parse_names_from_text(p.read_text(encoding="utf-8")))
                except Exception as exc:
                    tqdm.write(f"Failed to read extra list {p}: {exc}")
            else:
                tqdm.write(f"Extra list not found: {p}")

    names.update(load_patronymic_names())
    names.update(MANUAL_FEMALE_NAMES)

    normalized = {normalize_name(n) for n in names}
    normalized = {n for n in normalized if n}

    if not normalized:
        raise RuntimeError("No reference names could be loaded.")

    path.parent.mkdir(parents=True, exist_ok=True)
    if (not path.exists()) or (len(normalized) > len(existing)):
        path.write_text("\n".join(sorted(normalized)), encoding="utf-8")
        tqdm.write(f"Saved reference dictionary to {path.resolve()} (size {len(normalized)})")
    else:
        tqdm.write(f"Using existing dictionary at {path.resolve()} (size {len(normalized)})")

    return normalized


# -----------------------------------------------------------------------------
# Distance 1 index

ALPHABET = list(RUS_LETTERS) + [" "]


def _generate_neighbors(name: str) -> set[str]:
    variants: set[str] = set()
    L = len(name)

    # insertion
    for i in range(L + 1):
        prefix, suffix = name[:i], name[i:]
        for ch in ALPHABET:
            variants.add(prefix + ch + suffix)

    # deletion & substitution
    for i in range(L):
        prefix, suffix = name[:i], name[i + 1 :]
        variants.add(prefix + suffix)  # deletion
        for ch in ALPHABET:
            if ch != name[i]:
                variants.add(prefix + ch + suffix)

    # transposition
    for i in range(L - 1):
        if name[i] != name[i + 1]:
            variants.add(name[:i] + name[i + 1] + name[i] + name[i + 2 :])

    variants.discard(name)
    return variants


def build_neighbor_index(ref_names: Iterable[str]) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for ref in tqdm(ref_names, desc="Indexing reference names", unit="name"):
        if not ref:
            continue
        index[ref].add(ref)  # distance 0
        for neighbor in _generate_neighbors(ref):
            index[neighbor].add(ref)
    return index


# -----------------------------------------------------------------------------
# Matching logic

def wildcard_candidates(name: str, names_by_length: dict[int, list[str]]) -> list[str]:
    pattern = re.compile("^" + re.escape(name).replace(r"\*", ".") + "$")
    candidates = []
    for candidate in names_by_length.get(len(name), []):
        if pattern.match(candidate):
            candidates.append(candidate)
    return candidates


def classify_name(
    normalized_name: Optional[str],
    ref_names: set[str],
    neighbor_index: dict[str, set[str]],
    names_by_length: dict[int, list[str]],
    gender: Optional[str] = None,
) -> tuple[str, str, bool]:
    """Return corrected_name, class_label, is_name."""
    if not normalized_name:
        return "", "empty", False

    if "*" in normalized_name:
        candidates = wildcard_candidates(normalized_name, names_by_length)
        if len(candidates) == 1:
            cand = candidates[0]
            if _blocks_female_to_male(normalized_name, cand, gender):
                return normalized_name, "ambiguous", False
            return cand, "corrected", True
        if len(candidates) == 0:
            return normalized_name, "no_match", False
        return normalized_name, "ambiguous", False

    if normalized_name in ref_names:
        return normalized_name, "untouched", True

    candidates = neighbor_index.get(normalized_name, set())
    if len(candidates) == 1:
        cand = next(iter(candidates))
        if _blocks_female_to_male(normalized_name, cand, gender):
            return normalized_name, "ambiguous", False
        return cand, "corrected", True
    if len(candidates) > 1:
        return normalized_name, "ambiguous", False
    return normalized_name, "no_match", False


def _is_female_like(name: str) -> bool:
    return bool(name) and name[-1] in {"а", "я"}


def _gender_is_female(val: Optional[str]) -> bool:
    if val is None:
        return False
    v = str(val).strip().lower()
    return v in {"f", "ж", "female", "жен", "женский"}


def _blocks_female_to_male(source: str, target: str, gender: Optional[str]) -> bool:
    # If explicit gender says female, do not map to non-female-looking target.
    if _gender_is_female(gender) and not _is_female_like(target):
        return True
    # Fallback: avoid female-looking source -> non-female target
    return _is_female_like(source) and not _is_female_like(target)


# -----------------------------------------------------------------------------
# IO helpers

def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, engine="pyarrow")
    return pd.read_csv(path)


def _save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)


def _save_report(counts: dict, samples: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "class_distribution.json"
    samples_path = out_dir / "class_samples.json"
    report_path.write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")
    samples_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    tqdm.write(f"Saved reports to {out_dir.resolve()}")


# -----------------------------------------------------------------------------
# Main

def main():
    parser = argparse.ArgumentParser(description="Correct normalized names using DL distance=1.")
    parser.add_argument("--input", required=True, help="Parquet/CSV with normalized names")
    parser.add_argument(
        "--normalized-column",
        default="name_normalized",
        help="Column with normalized names",
    )
    parser.add_argument(
        "--gender-column",
        default=None,
        help="Optional gender column (F/Ж etc). If absent, logic falls back to name form.",
    )
    parser.add_argument(
        "--output",
        default="clean_data/clean_name/cleaned.parquet",
        help="Path to write corrected parquet",
    )
    parser.add_argument(
        "--dict-path",
        default=str(DEFAULT_DICT_PATH),
        help="Local path for reference dictionary (will be created if missing)",
    )
    parser.add_argument(
        "--report-dir",
        default="reports/name",
        help="Directory to write JSON reports",
    )
    parser.add_argument(
        "--extra-list",
        action="append",
        default=[],
        help="Path to file with additional names (one per line or CSV-like). Can be passed multiple times.",
    )
    parser.add_argument(
        "--extra-url",
        action="append",
        default=[],
        help="URL with additional names to merge. Can be passed multiple times.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Random sample per class for the report",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = _load_table(input_path)
    if args.normalized_column not in df.columns:
        raise KeyError(f"Column '{args.normalized_column}' not found in {input_path}")
    gender_series = None
    if args.gender_column:
        if args.gender_column in df.columns:
            gender_series = df[args.gender_column]
        else:
            tqdm.write(f"Gender column '{args.gender_column}' not found. Continue without it.")

    tqdm.write("Preparing reference dictionary...")
    extra_paths = [Path(p) for p in args.extra_list] if args.extra_list else []
    extra_urls = args.extra_url if args.extra_url else []
    ref_names = ensure_reference_names(Path(args.dict_path), extra_paths=extra_paths, extra_urls=extra_urls)
    names_by_length: dict[int, list[str]] = defaultdict(list)
    for n in ref_names:
        names_by_length[len(n)].append(n)

    neighbor_index = build_neighbor_index(ref_names)

    corrected, classes, flags = [], [], []
    genders_iter = gender_series.tolist() if gender_series is not None else [None] * len(df)
    for name, gender in tqdm(
        zip(df[args.normalized_column].tolist(), genders_iter),
        total=len(df),
        desc="Correcting",
        unit="name",
    ):
        corr, cls, ok = classify_name(name, ref_names, neighbor_index, names_by_length, gender)
        corrected.append(corr)
        classes.append(cls)
        flags.append(bool(ok))

    df["name_corrected"] = corrected
    df["class"] = classes
    df["is_name"] = flags

    out_path = Path(args.output)
    _save_table(df, out_path)
    tqdm.write(f"Saved corrected data to {out_path.resolve()}")

    counts = (
        pd.Series(classes)
        .value_counts()
        .sort_index()
        .to_dict()
    )

    samples: dict[str, list[dict]] = {}
    for cls in ["corrected", "untouched", "ambiguous", "empty", "no_match"]:
        subset = df[df["class"] == cls]
        if subset.empty:
            samples[cls] = []
            continue
        sample_df = subset.sample(n=min(args.sample_size, len(subset)), random_state=42)
        samples[cls] = (
            sample_df[[args.normalized_column, "name_corrected", "is_name"]]
            .rename(columns={args.normalized_column: "normalized_name"})
            .to_dict(orient="records")
        )

    _save_report(counts, samples, Path(args.report_dir))


if __name__ == "__main__":
    main()

