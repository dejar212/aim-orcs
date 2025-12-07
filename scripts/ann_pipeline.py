"""
End-to-end entity linkage pipeline with TF-IDF + nmslib HNSW + business rerank.

Steps implemented (per user brief):
- Load clean (finaldata) and raw (data/aim-2025-orcs) employees/orcs.
- Merge clean/raw by index with suffixes, fill missing with empty strings.
- Build token-tagged "soup" with weighting (INN/PAS repeated, YR_ prefix).
- Fit char-wb TF-IDF (3-5 grams, min_df=2, sublinear_tf=True).
- Build nmslib HNSW index on employee vectors, query k nearest for each orc.
- Hard filters (INN, birth year tolerance), fine-grained FIO similarity,
  gender penalty, rerank and threshold.
- Save top-1 matches to results/.

Progress bars:
- Stage progress in percent after each major step.
- tqdm bars for query batches and reranking to show live percent.

Optimized for Ryzen 7 5700X (8 cores) / 13GB+ RAM:
- dtype float32 to cut memory.
- multithreading for TF-IDF, nmslib index, batch queries.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import nmslib
import numpy as np
import pandas as pd
import joblib
import scipy.sparse as sp
from rapidfuzz import distance
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


BASE_COLUMNS = ["name", "surname", "fathername", "gender", "birthdate", "inn", "passport"]


@dataclass
class StageProgress:
    """Tracks coarse-grained progress as percentages."""

    weights: Dict[str, int]
    done: int = 0

    @property
    def total(self) -> int:
        return sum(self.weights.values())

    def bump(self, label: str) -> None:
        self.done += self.weights.get(label, 0)
        percent = self.done / self.total * 100
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Progress {percent:5.1f}% - {label}")


def load_and_merge(clean_path: str, raw_path: str) -> pd.DataFrame:
    """Load clean/raw parquet files, merge by index with suffixes, fill NaNs."""
    clean = pd.read_parquet(clean_path, columns=BASE_COLUMNS)
    raw = pd.read_parquet(raw_path, columns=BASE_COLUMNS)

    if len(clean) != len(raw):
        raise ValueError(
            f"Length mismatch: clean={len(clean)} raw={len(raw)}; cannot safe-align by index."
        )

    merged = pd.concat(
        [clean.add_suffix("_clean"), raw.add_suffix("_raw")],
        axis=1,
    )
    merged = merged.reset_index(drop=True).fillna("")

    # Ensure string dtype for downstream concatenations.
    for col in merged.columns:
        merged[col] = merged[col].astype(str)

    return merged


def compose_fio(df: pd.DataFrame, suffix: str) -> pd.Series:
    fio = (
        df[f"surname_{suffix}"]
        .str.cat([df[f"name_{suffix}"], df[f"fathername_{suffix}"]], sep=" ", na_rep="")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return fio


def extract_birth_year(df: pd.DataFrame) -> pd.Series:
    """Prefer clean birthdate, fall back to raw. Returns int Series (<=0 means unknown)."""
    clean_dt = pd.to_datetime(df["birthdate_clean"], errors="coerce")
    raw_dt = pd.to_datetime(df["birthdate_raw"], errors="coerce")
    year = clean_dt.dt.year.fillna(raw_dt.dt.year)
    year = year.fillna(-1).astype(int)
    return year


def pick_first(clean: pd.Series, raw: pd.Series) -> pd.Series:
    """Pick clean value when present, otherwise raw."""
    return clean.where(clean != "", raw)


def build_identifier_maps(employees: pd.DataFrame) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """Build lookup maps for INN and passport -> list of employee indices."""
    inn_map: Dict[str, List[int]] = {}
    pas_map: Dict[str, List[int]] = {}

    inn_vals = pick_first(employees["inn_clean"], employees["inn_raw"]).to_numpy()
    pas_vals = pick_first(employees["passport_clean"], employees["passport_raw"]).to_numpy()

    for idx, (inn, pas) in enumerate(zip(inn_vals, pas_vals)):
        if inn:
            inn_map.setdefault(inn, []).append(idx)
        if pas:
            pas_map.setdefault(pas, []).append(idx)
    return inn_map, pas_map


def select_best_candidate(cands: List[int], *, emp_year: np.ndarray, emp_gender: np.ndarray, orc_year: int, orc_gender: str) -> int:
    """Pick the best candidate among duplicates using birth year then gender."""
    if len(cands) == 1:
        return cands[0]

    best = cands[0]
    best_key = (10, 1)  # (year_diff_rank, gender_match_flag) larger gender better, smaller year_diff better
    for cid in cands:
        year_diff_rank = 10
        if orc_year > 0 and emp_year[cid] > 0:
            year_diff_rank = abs(orc_year - emp_year[cid])
        gender_match = 1 if (orc_gender and emp_gender[cid] and orc_gender == emp_gender[cid]) else 0
        key = (year_diff_rank, -gender_match)
        if key < best_key:
            best = cid
            best_key = key
    return best


def hard_match(
    orcs: pd.DataFrame,
    employees: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[int]]:
    """Deterministic matching by INN then passport. Returns matches and unmatched indices."""
    inn_map, pas_map = build_identifier_maps(employees)
    emp_year = extract_birth_year(employees).to_numpy()
    emp_gender = pick_first(employees["gender_clean"], employees["gender_raw"]).to_numpy()

    orc_inn = pick_first(orcs["inn_clean"], orcs["inn_raw"]).to_numpy()
    orc_pas = pick_first(orcs["passport_clean"], orcs["passport_raw"]).to_numpy()
    orc_year = extract_birth_year(orcs).to_numpy()
    orc_gender = pick_first(orcs["gender_clean"], orcs["gender_raw"]).to_numpy()

    results: List[Dict[str, object]] = []
    unmatched: List[int] = []

    for idx in tqdm(range(len(orcs)), desc="Hard match (INN/PAS)", unit="orc"):
        match_emp = -1
        status = "not_found"
        if orc_inn[idx]:
            cands = inn_map.get(orc_inn[idx], [])
            if cands:
                match_emp = select_best_candidate(
                    cands,
                    emp_year=emp_year,
                    emp_gender=emp_gender,
                    orc_year=orc_year[idx],
                    orc_gender=orc_gender[idx],
                )
                status = "hard_inn"
        if match_emp < 0 and orc_pas[idx]:
            cands = pas_map.get(orc_pas[idx], [])
            if cands:
                match_emp = select_best_candidate(
                    cands,
                    emp_year=emp_year,
                    emp_gender=emp_gender,
                    orc_year=orc_year[idx],
                    orc_gender=orc_gender[idx],
                )
                status = "hard_passport"

        if match_emp < 0:
            unmatched.append(idx)
            results.append(
                {
                    "orc_id": idx,
                    "employee_id": -1,
                    "status": status,
                    "final_score": -1.0,
                    "cosine_sim": 0.0,
                    "fio_score": 0.0,
                    "year_diff": None,
                }
            )
        else:
            year_diff = None
            if orc_year[idx] > 0 and emp_year[match_emp] > 0:
                year_diff = abs(orc_year[idx] - emp_year[match_emp])
            results.append(
                {
                    "orc_id": idx,
                    "employee_id": int(match_emp),
                    "status": status,
                    "final_score": 1.0,
                    "cosine_sim": 1.0,
                    "fio_score": 1.0,
                    "year_diff": year_diff,
                }
            )

    return pd.DataFrame(results), unmatched

def build_soup(df: pd.DataFrame) -> pd.Series:
    """Construct weighted text soup per record."""
    fio_clean = compose_fio(df, "clean")
    fio_raw = compose_fio(df, "raw")

    inn_primary = pick_first(df["inn_clean"], df["inn_raw"])
    passport_primary = pick_first(df["passport_clean"], df["passport_raw"])
    birth_year = extract_birth_year(df)

    inn_rep = np.where(
        inn_primary != "",
        "INN_" + inn_primary + " " + "INN_" + inn_primary + " " + "INN_" + inn_primary,
        "",
    )
    pas_rep = np.where(
        passport_primary != "",
        "PAS_" + passport_primary + " " + "PAS_" + passport_primary + " " + "PAS_" + passport_primary,
        "",
    )
    year_tok = np.where(birth_year > 0, "YR_" + birth_year.astype(str), "")

    soup = (
        fio_clean
        + " "
        + fio_raw
        + " "
        + inn_rep
        + " "
        + pas_rep
        + " "
        + pd.Series(year_tok, index=df.index)
    )
    soup = soup.str.replace(r"\s+", " ", regex=True).str.strip()
    return soup


def build_vectorizer(max_features: int | None = None) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.995,  # trim extremely frequent n-grams to save RAM
        sublinear_tf=True,
        dtype=np.float32,
        lowercase=True,
        max_features=max_features,
        norm="l2",
        stop_words=None,
        token_pattern=None,
    )


def build_hnsw_index(
    data: "scipy.sparse.csr_matrix",
    threads: int,
    ef_construction: int,
    M: int,
) -> nmslib.FloatIndex:
    index = nmslib.init(
        method="hnsw",
        space="cosinesimil_sparse",
        data_type=nmslib.DataType.SPARSE_VECTOR,
    )
    index.addDataPointBatch(data)
    index.createIndex(
        {
            "post": 2,
            "M": M,
            "efConstruction": ef_construction,
            "indexThreadQty": threads,
        },
        print_progress=True,
    )
    index.setQueryTimeParams({"efSearch": max(ef_construction // 2, 50)})
    return index


def batched_knn(
    index: nmslib.FloatIndex,
    queries: "scipy.sparse.csr_matrix",
    k: int,
    threads: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Query index in batches with progress, returns (ids, similarities)."""
    n = queries.shape[0]
    ids = np.full((n, k), -1, dtype=np.int32)
    sims = np.zeros((n, k), dtype=np.float32)

    for start in tqdm(range(0, n, batch_size), desc="Index query (k-NN)", unit="batch"):
        end = min(start + batch_size, n)
        batch_res = index.knnQueryBatch(
            queries[start:end],
            k=k,
            num_threads=threads,
        )
        for i, (nbr_ids, nbr_dists) in enumerate(batch_res):
            row = start + i
            if len(nbr_ids) == 0:
                continue
            sim_row = 1.0 - np.array(nbr_dists, dtype=np.float32)
            ids[row, : len(nbr_ids)] = nbr_ids
            sims[row, : len(sim_row)] = sim_row
    return ids, sims


def rerank(
    orcs: pd.DataFrame,
    employees: pd.DataFrame,
    neighbor_ids: np.ndarray,
    neighbor_sims: np.ndarray,
    *,
    year_tolerance: int,
    threshold: float,
    orc_indices: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Apply hard filters + fine-grained scoring."""
    results: List[Dict[str, object]] = []

    emp_fio = compose_fio(employees, "clean").to_numpy()
    emp_inn = pick_first(employees["inn_clean"], employees["inn_raw"]).to_numpy()
    emp_gender = pick_first(employees["gender_clean"], employees["gender_raw"]).to_numpy()
    emp_year = extract_birth_year(employees).to_numpy()

    orc_fio = compose_fio(orcs, "clean").to_numpy()
    orc_inn = pick_first(orcs["inn_clean"], orcs["inn_raw"]).to_numpy()
    orc_gender = pick_first(orcs["gender_clean"], orcs["gender_raw"]).to_numpy()
    orc_year = extract_birth_year(orcs).to_numpy()

    for local_idx in tqdm(range(len(orcs)), desc="Reranking + hard filters", unit="orc"):
        idx = orc_indices[local_idx] if orc_indices is not None else local_idx
        best_score = -1.0
        best_emp = -1
        best_cos = 0.0
        best_fio = 0.0
        best_year_diff = None
        candidates = neighbor_ids[local_idx]
        sims = neighbor_sims[local_idx]

        for cid, cos_sim in zip(candidates, sims):
            if cid < 0:
                continue
            inn_o = orc_inn[idx]
            inn_e = emp_inn[cid]
            if inn_o and inn_e and inn_o != inn_e:
                continue

            by_o = orc_year[idx]
            by_e = emp_year[cid]
            year_diff = None
            if by_o > 0 and by_e > 0:
                year_diff = abs(by_o - by_e)
                if year_diff > year_tolerance:
                    continue

            fio_score = float(distance.Levenshtein.normalized_similarity(orc_fio[idx], emp_fio[cid]))

            # Birth bonus.
            birth_bonus = 0.0
            if year_diff is not None:
                if year_diff == 0:
                    birth_bonus = 0.05
                elif year_diff <= 1:
                    birth_bonus = 0.02
                elif year_diff <= 2:
                    birth_bonus = 0.01

            score = 0.55 * cos_sim + 0.45 * fio_score + birth_bonus

            gender_o = orc_gender[idx]
            gender_e = emp_gender[cid]
            if gender_o and gender_e and gender_o != gender_e:
                score *= 0.75  # stronger penalty for precision

            if score > best_score:
                best_score = score
                best_emp = cid
                best_cos = float(cos_sim)
                best_fio = fio_score
                best_year_diff = year_diff

        if best_score < threshold or best_emp < 0:
            results.append(
                {
                    "orc_id": idx,
                    "employee_id": -1,
                    "status": "not_found",
                    "final_score": float(best_score),
                    "cosine_sim": float(best_cos),
                    "fio_score": float(best_fio),
                    "year_diff": best_year_diff,
                }
            )
        else:
            results.append(
                {
                    "orc_id": idx,
                    "employee_id": int(best_emp),
                    "status": "matched",
                    "final_score": float(best_score),
                    "cosine_sim": float(best_cos),
                    "fio_score": float(best_fio),
                    "year_diff": best_year_diff,
                }
            )

    return pd.DataFrame(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TF-IDF + HNSW linkage with reranking.")
    parser.add_argument("--k", type=int, default=50, help="Neighbors per orc to retrieve from ANN.")
    parser.add_argument("--threads", type=int, default=8, help="Threads for TF-IDF and HNSW.")
    parser.add_argument("--ef-construction", type=int, default=200, help="HNSW efConstruction.")
    parser.add_argument("--M", type=int, default=32, help="HNSW M (higher -> better recall, more RAM).")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for ANN queries.")
    parser.add_argument("--year-tolerance", type=int, default=1, help="Max birth year diff before reject (precision).")
    parser.add_argument("--threshold", type=float, default=0.6, help="Min final score to accept match (precision).")
    parser.add_argument(
        "--max-features",
        type=int,
        default=750_000,
        help="Cap TF-IDF vocab to save RAM (char 3-5g). Lower if MemoryError occurs.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="temp/ann_cache",
        help="Folder for cached vectorizer and TF-IDF matrices.",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Load cached vectorizer/TF-IDF if available to skip refit.",
    )
    parser.add_argument(
        "--save-cache",
        action="store_true",
        help="Save vectorizer and TF-IDF matrices after computation.",
    )
    parser.add_argument(
        "--limit-employees",
        type=int,
        default=None,
        help="Optional subset of employees for quicker trial run.",
    )
    parser.add_argument(
        "--limit-orcs",
        type=int,
        default=None,
        help="Optional subset of orcs for quicker trial run.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to store outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    progress = StageProgress(
        weights={
            "load_merge": 5,
            "soup": 10,
            "vectorizer_fit": 20,
            "vectorizer_transform": 20,
            "index_build": 20,
            "query": 10,
            "rerank": 15,
        }
    )

    print("Loading and merging datasets...")
    employees = load_and_merge("finaldata/employees_final.parquet", "data/aim-2025-orcs/employees.parquet")
    orcs = load_and_merge("finaldata/orcs_final.parquet", "data/aim-2025-orcs/orcs.parquet")

    if args.limit_employees:
        employees = employees.head(args.limit_employees).reset_index(drop=True)
    if args.limit_orcs:
        orcs = orcs.head(args.limit_orcs).reset_index(drop=True)
    progress.bump("load_merge")

    print("Building soups (token-tagged, weighted)...")
    employees["soup"] = build_soup(employees)
    orcs["soup"] = build_soup(orcs)
    progress.bump("soup")

    print("Running hard deterministic matches (INN/PAS)...")
    hard_df, unmatched_orc_indices = hard_match(orcs, employees)
    print(f"Hard-matched: {len(orcs) - len(unmatched_orc_indices)} / {len(orcs)}")

    cache_vec = os.path.join(args.cache_dir, "vectorizer.joblib")
    cache_emp = os.path.join(args.cache_dir, "emp_tfidf.npz")
    cache_orc = os.path.join(args.cache_dir, "orc_tfidf.npz")

    vectorizer = None
    emp_tfidf = None
    orc_tfidf = None

    if args.use_cache and all(os.path.exists(p) for p in [cache_vec, cache_emp, cache_orc]):
        print("Loading cached vectorizer and TF-IDF matrices...")
        vectorizer = joblib.load(cache_vec)
        emp_tfidf = sp.load_npz(cache_emp)
        orc_tfidf = sp.load_npz(cache_orc)
        if emp_tfidf.shape[0] != len(employees) or orc_tfidf.shape[0] != len(orcs):
            print("Cache shape mismatch with current data; recomputing TF-IDF.")
            vectorizer = None
            emp_tfidf = None
            orc_tfidf = None
        else:
            progress.bump("vectorizer_fit")
            progress.bump("vectorizer_transform")

    if vectorizer is None or emp_tfidf is None or orc_tfidf is None:
        print("Fitting TF-IDF vectorizer on combined corpus...")
        vectorizer = build_vectorizer(max_features=args.max_features)
        combined_corpus = pd.concat([employees["soup"], orcs["soup"]], ignore_index=True)
        try:
            vectorizer.fit(combined_corpus)
        except MemoryError:
            print(
                "MemoryError during TF-IDF fit. "
                "Try lowering --max-features (e.g., 400000) or increasing available RAM."
            )
            raise
        del combined_corpus
        progress.bump("vectorizer_fit")

        print("Transforming employees and orcs soups...")
        emp_tfidf = vectorizer.transform(employees["soup"])
        orc_tfidf = vectorizer.transform(orcs["soup"])
        progress.bump("vectorizer_transform")

        if args.save_cache:
            print("Saving vectorizer and TF-IDF matrices to cache...")
            joblib.dump(vectorizer, cache_vec)
            sp.save_npz(cache_emp, emp_tfidf)
            sp.save_npz(cache_orc, orc_tfidf)

    ann_matches = pd.DataFrame(columns=["orc_id", "employee_id", "status", "final_score", "cosine_sim", "fio_score", "year_diff"])
    if len(unmatched_orc_indices) > 0:
        print("Building HNSW index (nmslib)...")
        index = build_hnsw_index(
            data=emp_tfidf,
            threads=args.threads,
            ef_construction=args.ef_construction,
            M=args.M,
        )
        progress.bump("index_build")

        print(f"Querying ANN index for {len(unmatched_orc_indices)} unmatched orcs (k={args.k})...")
        orc_tfidf_unmatched = orc_tfidf[unmatched_orc_indices]
        neighbor_ids, neighbor_sims = batched_knn(
            index=index,
            queries=orc_tfidf_unmatched,
            k=args.k,
            threads=args.threads,
            batch_size=args.batch_size,
        )
        progress.bump("query")

        print("Reranking candidates with hard filters and string similarity...")
        orcs_subset = orcs.loc[unmatched_orc_indices].reset_index(drop=True)
        ann_matches = rerank(
            orcs_subset,
            employees,
            neighbor_ids,
            neighbor_sims,
            year_tolerance=args.year_tolerance,
            threshold=args.threshold,
            orc_indices=unmatched_orc_indices,
        )
        progress.bump("rerank")
    else:
        progress.bump("index_build")
        progress.bump("query")
        progress.bump("rerank")

    # Merge hard matches with ANN fallback (ANN overrides not_found rows).
    final_df = hard_df.set_index("orc_id")
    if not ann_matches.empty:
        for _, row in ann_matches.iterrows():
            final_df.loc[int(row["orc_id"])] = row.to_dict()
    matches = final_df.reset_index(drop=True).sort_values("orc_id")

    out_parquet = os.path.join(args.output_dir, "orc_matches_top1.parquet")
    out_csv = os.path.join(args.output_dir, "orc_matches_top1.csv")
    matches.to_parquet(out_parquet, index=False)
    matches.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved results to {out_parquet} and {out_csv}")

    # Produce submission-style file without nulls.
    submission = pd.DataFrame(
        {
            "id": matches["orc_id"].astype(int),
            "orig_index": matches["employee_id"].fillna(-1).astype(int),
        }
    )
    sub_parquet = os.path.join(args.output_dir, "submission_from_ann.parquet")
    sub_csv = os.path.join(args.output_dir, "submission_from_ann.csv")
    submission.to_parquet(sub_parquet, index=False)
    submission.to_csv(sub_csv, index=False)
    print(f"Saved submission to {sub_parquet} and {sub_csv}")


if __name__ == "__main__":
    main()

