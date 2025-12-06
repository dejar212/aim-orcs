"""Fuzzy correction for normalized patronymics using patronymics dictionary.

Dictionary source: patronymics.csv (Name, Male Patronymic, Female Patronymic)
Flow per value:
- empty -> class 'empty'
- '*' present -> wildcard match vs dictionary; single -> corrected, none -> no_match, many -> ambiguous
- exact dictionary hit -> 'untouched'
- DL distance=1 vs dictionary (on-the-fly neighbor gen) -> single -> corrected, many -> ambiguous
- ending fix (distance<=1 to allowed patronymic endings) -> single -> corrected; multiple insert-type -> add '*'
- else -> no_match
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from tqdm import tqdm

from normalization import RUS_LETTERS
from patronymic_normalization import normalize_patronymic

PATRONYMICS_CSV = Path("patronymics.csv")
DEFAULT_DICT_PATH = Path("data/reference/patronymics.txt")

PATRONYMIC_ENDINGS = [
    "ович",
    "евич",
    "ич",
    "овна",
    "евна",
    "ична",
]

ALPHABET = list(RUS_LETTERS) + [" "]


def load_patronymic_dict(csv_path: Path = PATRONYMICS_CSV) -> set[str]:
    names: set[str] = set()
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        cols = df.columns.tolist()
        for col in cols[1:]:  # skip base name
            names.update(df[col].dropna().astype(str).tolist())
    return {n for n in names if n}


def ensure_reference(path: Path = DEFAULT_DICT_PATH) -> set[str]:
    existing: set[str] = set()
    if path.exists():
        existing = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    names = set(existing)
    names.update(load_patronymic_dict())

    normalized = {normalize_patronymic(n) for n in names}
    normalized = {n for n in normalized if n}
    if not normalized:
        raise RuntimeError("No patronymics could be loaded.")

    if (not path.exists()) or (len(normalized) > len(existing)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(sorted(normalized)), encoding="utf-8")
        tqdm.write(f"Saved patronymic dictionary to {path.resolve()} (size {len(normalized)})")
    else:
        tqdm.write(f"Using existing patronymic dictionary at {path.resolve()} (size {len(normalized)})")
    return normalized


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
        variants.add(prefix + suffix)
        for ch in ALPHABET:
            if ch != name[i]:
                variants.add(prefix + ch + suffix)
    # transposition
    for i in range(L - 1):
        if name[i] != name[i + 1]:
            variants.add(name[:i] + name[i + 1] + name[i] + name[i + 2 :])
    variants.discard(name)
    return variants


def wildcard_candidates(name: str, names_by_length: dict[int, list[str]]) -> list[str]:
    pattern = re.compile("^" + re.escape(name).replace(r"\*", ".") + "$")
    candidates = []
    for candidate in names_by_length.get(len(name), []):
        if pattern.match(candidate):
            candidates.append(candidate)
    return candidates


def distance1_op(a: str, b: str):
    if a == b:
        return ("equal", -1)
    if abs(len(a) - len(b)) > 1:
        return None
    la, lb = len(a), len(b)
    if la == lb:
        mism = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        if len(mism) == 1:
            return ("replace", mism[0])
        if len(mism) == 2 and mism[0] + 1 == mism[1] and a[mism[0]] == b[mism[1]] and a[mism[1]] == b[mism[0]]:
            return ("transpose", mism[0])
        return None
    if la + 1 == lb:
        i = 0
        while i < la and a[i] == b[i]:
            i += 1
        if a[i:] == b[i + 1 :]:
            return ("insert", i)
        return None
    if la == lb + 1:
        i = 0
        while i < lb and a[i] == b[i]:
            i += 1
        if a[i + 1 :] == b[i:]:
            return ("delete", i)
        return None
    return None


def correct_ending(token: str) -> tuple[str, Optional[str]]:
    if not token or len(token) <= 2:
        return token, None
    candidates = []
    token_len = len(token)
    for ending in PATRONYMIC_ENDINGS:
        k = len(ending)
        for obs_len in {k - 1, k, k + 1}:
            if obs_len <= 0 or obs_len > token_len:
                continue
            obs = token[-obs_len:]
            op = distance1_op(obs, ending)
            if op is None:
                continue
            dist = 0 if op[0] == "equal" else 1
            prefix = token[:-obs_len] if obs_len <= token_len else ""
            candidates.append((dist, op, ending, obs, prefix))
    if not candidates:
        return token, None
    min_dist = min(c[0] for c in candidates)
    candidates = [c for c in candidates if c[0] == min_dist]
    if len(candidates) == 1:
        _, _, ending, _, prefix = candidates[0]
        return prefix + ending, ending
    ops = {c[1][0] for c in candidates}
    if ops == {"insert"}:
        positions = {c[1][1] for c in candidates}
        pos = positions.pop() if len(positions) == 1 else 0
        _, _, _, obs, prefix = candidates[0]
        wildcard_ending = obs[:pos] + "*" + obs[pos:]
        return prefix + wildcard_ending, None
    return token, None


def classify_token(
    token: Optional[str],
    ref_names: set[str],
    names_by_length: dict[int, list[str]],
) -> tuple[str, str, bool]:
    if not token:
        return "", "empty", False
    if "*" in token:
        cands = wildcard_candidates(token, names_by_length)
        if len(cands) == 1:
            return cands[0], "corrected", True
        if len(cands) == 0:
            return token, "no_match", False
        return token, "ambiguous", False
    if token in ref_names:
        return token, "untouched", True
    cands = {cand for cand in _generate_neighbors(token) if cand in ref_names}
    if len(cands) == 1:
        return next(iter(cands)), "corrected", True
    if len(cands) > 1:
        return token, "ambiguous", False
    fixed, applied = correct_ending(token)
    if fixed != token:
        if applied is None:
            return fixed, "ambiguous", False
        return fixed, "corrected", True
    return token, "no_match", False


def classify_value(
    normalized: Optional[str],
    ref_names: set[str],
    names_by_length: dict[int, list[str]],
) -> tuple[str, str, bool]:
    if not normalized:
        return "", "empty", False
    parts = normalized.split()
    corrected_parts = []
    class_parts = []
    is_flags = []
    for part in parts:
        corr, cls, ok = classify_token(part, ref_names, names_by_length)
        corrected_parts.append(corr)
        class_parts.append(cls)
        is_flags.append(ok)
    priority = {"corrected": 3, "ambiguous": 2, "no_match": 1, "untouched": 0, "empty": -1}
    best_cls = max(class_parts, key=lambda c: priority.get(c, -2))
    is_ok = all(is_flags)
    return " ".join(corrected_parts), best_cls, is_ok


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, engine="pyarrow")
    return pd.read_csv(path)


def _save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)


def _save_report(counts: dict, samples: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "class_distribution.json").write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "class_samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    tqdm.write(f"Saved reports to {out_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Correct normalized patronymics.")
    parser.add_argument("--input", required=True, help="Parquet/CSV with normalized patronymics")
    parser.add_argument(
        "--normalized-column",
        default="patronymic_normalized",
        help="Column with normalized patronymics",
    )
    parser.add_argument(
        "--output",
        default="clean_data/clean_patronymic/cleaned.parquet",
        help="Path to write corrected parquet",
    )
    parser.add_argument(
        "--dict-path",
        default=str(DEFAULT_DICT_PATH),
        help="Local path for patronymic dictionary (will be created if missing)",
    )
    parser.add_argument(
        "--report-dir",
        default="reports/patronymic",
        help="Directory to write JSON reports",
    )
    parser.add_argument("--sample-size", type=int, default=20)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = _load_table(input_path)
    if args.normalized_column not in df.columns:
        raise KeyError(f"Column '{args.normalized_column}' not found in {input_path}")

    ref_names = ensure_reference(Path(args.dict_path))
    names_by_length: dict[int, list[str]] = defaultdict(list)
    for n in ref_names:
        names_by_length[len(n)].append(n)

    corrected, classes, flags = [], [], []
    for token in tqdm(df[args.normalized_column].tolist(), desc="Correcting patronymics", unit="patr"):
        corr, cls, ok = classify_value(token, ref_names, names_by_length)
        corrected.append(corr)
        classes.append(cls)
        flags.append(bool(ok))

    df["patronymic_corrected"] = corrected
    df["class"] = classes
    df["is_patronymic"] = flags

    out_path = Path(args.output)
    _save_table(df, out_path)
    tqdm.write(f"Saved corrected data to {out_path.resolve()}")

    counts = pd.Series(classes).value_counts().sort_index().to_dict()
    samples: dict[str, list[dict]] = {}
    for cls in ["corrected", "untouched", "ambiguous", "empty", "no_match"]:
        subset = df[df["class"] == cls]
        if subset.empty:
            samples[cls] = []
            continue
        sample_df = subset.sample(n=min(args.sample_size, len(subset)), random_state=42)
        samples[cls] = (
            sample_df[[args.normalized_column, "patronymic_corrected", "is_patronymic"]]
            .rename(columns={args.normalized_column: "normalized_patronymic"})
            .to_dict(orient="records")
        )
    _save_report(counts, samples, Path(args.report_dir))


if __name__ == "__main__":
    main()

