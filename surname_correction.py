"""Fuzzy correction for normalized surnames with ending-aware logic.

Flow per surname part (hyphen split handled at normalization):
- empty -> class 'empty'
- '*' present -> wildcard match vs dictionary; if single hit -> corrected else ambiguous/no_match
- exact dictionary hit -> 'untouched'
- distance=1 vs dictionary -> if single hit -> 'corrected', multiple -> 'ambiguous'
- ending fix: if suffix within 1 edit of allowed endings list -> single candidate -> corrected;
  if multiple insertion-type candidates -> insert '*' at missing position; else ambiguous
- else -> 'no_match'
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Optional
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from normalization import RUS_LETTERS
from surname_normalization import normalize_surname

ENDINGS = [
    "ов", "ев", "ёв", "ин", "ын", "ский", "ская", "цкий", "цкая", "их", "ых", "ово", "аго", "ого",
    "енко", "ко", "ук", "юк", "ак", "як", "ич", "ик", "чик", "ка", "онак", "ёнак", "ня",
    "швили", "дзе", "ури", "ава", "уа", "иа", "ни", "ли",
    "ян", "янц", "уни",
    "заде", "ли", "лы",
    "улы",
]

DEFAULT_DICT_PATH = Path("russian_surnames.txt")


def download_text(url: str, timeout: int = 15) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def parse_names_from_text(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            names: list[str] = []
            for item in data:
                if isinstance(item, dict):
                    if "name" in item:
                        names.append(str(item["name"]))
                elif isinstance(item, str):
                    names.append(item)
            if names:
                return names
        elif isinstance(data, dict) and "name" in data:
            return [str(data["name"])]
    except json.JSONDecodeError:
        pass
    names: list[str] = []
    for line in raw.splitlines():
        for token in re.split(r"[;,]", line):
            token = token.strip()
            if token:
                names.append(token)
    return names


def ensure_reference_surnames(path: Path = DEFAULT_DICT_PATH, extra_paths=None, extra_urls=None) -> set[str]:
    existing: set[str] = set()
    if path.exists():
        existing = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    names: set[str] = set(existing)

    def _extend_from_urls(urls):
        if not urls:
            return
        for url in urls:
            try:
                tqdm.write(f"Downloading surnames from {url}")
                raw = download_text(url)
                names.update(parse_names_from_text(raw))
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                tqdm.write(f"Failed to download {url}: {exc}")

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

    # Always include local russian_surnames.txt content if present
    if path.exists():
        names.update(existing)

    normalized = {normalize_surname(n) for n in names}
    normalized = {n for n in normalized if n}

    if not normalized:
        raise RuntimeError("No reference surnames could be loaded.")

    if (not path.exists()) or (len(normalized) > len(existing)):
        path.write_text("\n".join(sorted(normalized)), encoding="utf-8")
        tqdm.write(f"Saved surname dictionary to {path.resolve()} (size {len(normalized)})")
    else:
        tqdm.write(f"Using existing surname dictionary at {path.resolve()} (size {len(normalized)})")

    return normalized


# Distance 1 helper for endings ------------------------------------------------

def distance1_op(a: str, b: str) -> Optional[tuple[str, int]]:
    """Return (op, pos) if distance <=1 else None.
    op in {"equal","replace","insert","delete","transpose"}.
    pos is index in the longer/first string depending on op; used only for insert.
    """
    if a == b:
        return ("equal", -1)
    if abs(len(a) - len(b)) > 1:
        return None
    la, lb = len(a), len(b)
    # same length: replace or transpose
    if la == lb:
        mism = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        if len(mism) == 1:
            return ("replace", mism[0])
        if len(mism) == 2 and mism[0] + 1 == mism[1] and a[mism[0]] == b[mism[1]] and a[mism[1]] == b[mism[0]]:
            return ("transpose", mism[0])
        return None
    # insertion: b longer by 1 (a missing char)
    if la + 1 == lb:
        i = 0
        while i < la and a[i] == b[i]:
            i += 1
        # insert position i (in b)
        if a[i:] == b[i + 1 :]:
            return ("insert", i)
        return None
    # deletion: a longer by 1
    if la == lb + 1:
        i = 0
        while i < lb and a[i] == b[i]:
            i += 1
        if a[i + 1 :] == b[i:]:
            return ("delete", i)
        return None
    return None


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


# Matching logic --------------------------------------------------------------

def wildcard_candidates(name: str, names_by_length: dict[int, list[str]]) -> list[str]:
    pattern = re.compile("^" + re.escape(name).replace(r"\*", ".") + "$")
    candidates = []
    for candidate in names_by_length.get(len(name), []):
        if pattern.match(candidate):
            candidates.append(candidate)
    return candidates


def correct_ending(token: str) -> tuple[str, Optional[str]]:
    """Return (corrected_token, applied_ending_or_None_if_none)."""
    if not token or len(token) <= 2:
        return token, None
    candidates = []
    max_len = max(len(e) for e in ENDINGS)
    min_len = min(len(e) for e in ENDINGS)
    token_len = len(token)
    for ending in ENDINGS:
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
    # prefer minimal distance
    min_dist = min(c[0] for c in candidates)
    candidates = [c for c in candidates if c[0] == min_dist]
    if len(candidates) == 1:
        _, _, ending, _, prefix = candidates[0]
        return prefix + ending, ending
    # Multiple candidates
    ops = {c[1][0] for c in candidates}
    if ops == {"insert"}:
        # same insert position?
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
    # ending-based fix
    fixed, applied = correct_ending(token)
    if fixed != token:
        if applied is None:
            return fixed, "ambiguous", False
        return fixed, "corrected", True
    # Heuristic: missing final 'в' or missing 'о' in 'кв' endings (e.g., корпало->корпалов, скердо->скердов, тиллекв->тиллеков)
    if token.endswith("о"):
        candidate = token + "в"
        if candidate in ref_names:
            return candidate, "corrected", True
    if token.endswith("до"):
        candidate = token + "в"
        if candidate in ref_names:
            return candidate, "corrected", True
    if token.endswith("кв") and len(token) > 3:
        candidate = token[:-2] + "ков"
        if candidate in ref_names:
            return candidate, "corrected", True
    return token, "no_match", False


def classify_surname(
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
    # Overall class: if any corrected -> corrected, else if all untouched -> untouched, else ambiguous/no_match/empty propagate max severity
    # Priority: corrected > ambiguous > no_match > untouched > empty
    priority = {"corrected": 3, "ambiguous": 2, "no_match": 1, "untouched": 0, "empty": -1}
    best_cls = max(class_parts, key=lambda c: priority.get(c, -2))
    is_ok = all(is_flags)
    return " ".join(corrected_parts), best_cls, is_ok


# IO helpers ------------------------------------------------------------------

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
    parser = argparse.ArgumentParser(description="Correct normalized surnames.")
    parser.add_argument("--input", required=True, help="Parquet/CSV with normalized surnames")
    parser.add_argument(
        "--normalized-column",
        default="surname_normalized",
        help="Column with normalized surnames",
    )
    parser.add_argument(
        "--output",
        default="clean_data/clean_surname/cleaned.parquet",
        help="Path to write corrected parquet",
    )
    parser.add_argument(
        "--dict-path",
        default=str(DEFAULT_DICT_PATH),
        help="Local path for surname dictionary (will be created if missing)",
    )
    parser.add_argument(
        "--report-dir",
        default="reports/surname",
        help="Directory to write JSON reports",
    )
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument(
        "--extra-list",
        action="append",
        default=[],
        help="Path to file with additional surnames; can repeat.",
    )
    parser.add_argument(
        "--extra-url",
        action="append",
        default=[],
        help="URL with additional surnames; can repeat.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = _load_table(input_path)
    if args.normalized_column not in df.columns:
        raise KeyError(f"Column '{args.normalized_column}' not found in {input_path}")

    extra_paths = [Path(p) for p in args.extra_list] if args.extra_list else []
    extra_urls = args.extra_url if args.extra_url else []
    ref_names = ensure_reference_surnames(Path(args.dict_path), extra_paths=extra_paths, extra_urls=extra_urls)
    names_by_length: dict[int, list[str]] = defaultdict(list)
    for n in ref_names:
        names_by_length[len(n)].append(n)

    corrected, classes, flags = [], [], []
    for token in tqdm(df[args.normalized_column].tolist(), desc="Correcting surnames", unit="surname"):
        corr, cls, ok = classify_surname(token, ref_names, names_by_length)
        corrected.append(corr)
        classes.append(cls)
        flags.append(bool(ok))

    df["surname_corrected"] = corrected
    df["class"] = classes
    df["is_surname"] = flags

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
            sample_df[[args.normalized_column, "surname_corrected", "is_surname"]]
            .rename(columns={args.normalized_column: "normalized_surname"})
            .to_dict(orient="records")
        )
    _save_report(counts, samples, Path(args.report_dir))


if __name__ == "__main__":
    main()

