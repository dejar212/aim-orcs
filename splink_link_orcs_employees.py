#!/usr/bin/env python
"""
Pipeline для link-only сопоставления орков и сотрудников через Splink (DuckDB backend).

Запуск:
    python splink_link_orcs_employees.py

Результат:
    В каталоге results/<timestamp>/ будут сохранены suspected_orcs.csv и tunable_params.txt.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import phonetics
from splink.duckdb.comparison_library import (
    exact_match,
    jaro_winkler_at_thresholds,
)
from splink.duckdb.linker import DuckDBLinker

# Пути к данным и каталог результатов
DATA_DIR = Path("finaldata")
RESULTS_DIR = Path("results")

# Файлы-источники
ORCS_PATH = DATA_DIR / "orcs_final.parquet"
EMPLOYEES_PATH = DATA_DIR / "employees_final.parquet"

# Смещение, чтобы уникальные ID сотрудников не пересекались с ID орков
EMPLOYEE_ID_OFFSET = 10_000_000


def _clean_str_series(series: pd.Series) -> pd.Series:
    """Приводит строки к нижнему регистру, убирает пробелы и пустые значения."""
    cleaned = (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
        .replace({"": pd.NA})
    )
    return cleaned


def load_and_prepare(path: Path, dataset_name: str, id_offset: int = 0) -> pd.DataFrame:
    """Загрузка parquet и приведение колонок к единому стандарту."""
    base_columns = [
        "name",
        "surname",
        "fathername",
        "birthdate",
        "inn",
        "passport",
        "name_normalized",
        "surname_normalized",
        "patronymic_normalized",
    ]

    df = pd.read_parquet(path, columns=base_columns, engine="pyarrow")
    df = df.reset_index(drop=True)

    # Базовые ID
    df["row_id"] = df.index.astype("int64")
    df["unique_id"] = df["row_id"] + id_offset
    df["source_dataset"] = dataset_name

    # Нормализованные ФИО
    df["first_name"] = _clean_str_series(
        df["name_normalized"].fillna(df["name"])
    )
    df["surname_std"] = _clean_str_series(
        df["surname_normalized"].fillna(df["surname"])
    )
    def _safe_soundex(val: object) -> object:
        if not isinstance(val, str):
            return pd.NA
        ascii_only = "".join(ch for ch in val if ch.isascii())
        if not ascii_only:
            return pd.NA
        try:
            return phonetics.soundex(ascii_only)
        except Exception:
            return pd.NA

    df["surname_soundex"] = df["surname_std"].apply(_safe_soundex)
    df["patronymic"] = _clean_str_series(
        df["patronymic_normalized"].fillna(df["fathername"])
    )

    # Дата рождения как строка YYYY-MM-DD
    df["dob"] = pd.to_datetime(df["birthdate"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )

    # Документы и ИНН оставляем строками
    df["inn"] = _clean_str_series(df["inn"])
    df["passport"] = _clean_str_series(df["passport"])

    # Префикс фамилии для блокировки
    df["surname_prefix4"] = df["surname_std"].str.slice(0, 4)

    # Полное имя (для ручного анализа, не обязательно в модели)
    df["full_name"] = (
        df[["surname_std", "first_name", "patronymic"]]
        .fillna("")
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace({"": pd.NA})
    )

    return df


def build_settings(blocking_rules: List[str]) -> Dict:
    """Формирует settings_dict для Splink."""
    dob_comparison = {
        "output_column_name": "dob",
        "comparison_levels": [
            {
                "sql_condition": "dob_l IS NULL OR dob_r IS NULL",
                "label_for_charts": "dob is null",
                "is_null_level": True,
            },
            {
                "sql_condition": "dob_l = dob_r",
                "label_for_charts": "dob exact",
            },
            {
                "sql_condition": "abs(date_diff('day', cast(dob_l as date), cast(dob_r as date))) <= 1",
                "label_for_charts": "dob within 1 day",
            },
            {"sql_condition": "ELSE", "label_for_charts": "dob all other"},
        ],
    }

    settings = {
        "link_type": "link_only",
        "unique_id_column_name": "unique_id",
        "source_dataset_column_name": "source_dataset",
        "blocking_rules_to_generate_predictions": blocking_rules,
        "comparisons": [
            jaro_winkler_at_thresholds(
                "surname_std",
                [0.99, 0.95, 0.88],
                term_frequency_adjustments=False,
            ),
            jaro_winkler_at_thresholds(
                "first_name",
                [0.99, 0.95, 0.88],
                term_frequency_adjustments=False,
            ),
            jaro_winkler_at_thresholds(
                "patronymic",
                [0.99, 0.95, 0.88],
                term_frequency_adjustments=False,
            ),
            dob_comparison,
            exact_match("inn"),
            exact_match("passport"),
        ],
    }
    return settings


def main(match_threshold: Optional[float] = None) -> None:
    # Создаем каталог результатов с таймстампом
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"splink_results_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Загружаем данные
    print("Загружаем данные...")
    orcs_df = load_and_prepare(ORCS_PATH, "orcs", id_offset=0)
    employees_df = load_and_prepare(EMPLOYEES_PATH, "employees", id_offset=EMPLOYEE_ID_OFFSET)

    # Карты ID для возврата к исходным индексам
    orc_id_map = orcs_df.set_index("unique_id")["row_id"].to_dict()
    emp_id_map = employees_df.set_index("unique_id")["row_id"].to_dict()

    # Блокировочные правила
    blocking_rules = [
        "l.dob = r.dob",
        "l.inn IS NOT NULL AND r.inn IS NOT NULL AND l.inn = r.inn",
        "l.surname_std = r.surname_std",
        "substr(l.surname_std, 1, 4) = substr(r.surname_std, 1, 4)",
        "l.surname_soundex = r.surname_soundex",
    ]

    settings = build_settings(blocking_rules)

    print("Строим DuckDBLinker...")
    linker = DuckDBLinker(
        [orcs_df, employees_df],
        settings,
        input_table_aliases=["orcs", "employees"],
        connection=str(run_dir / "splink_db.duckdb"),
        set_up_basic_logging=False,
    )

    # Оценка u с случайной выборкой
    print("Оцениваем u при помощи случайной выборки...")
    linker.estimate_u_using_random_sampling(max_pairs=int(1e6))

    # EM по строгим блокировкам
    print("Запускаем EM...")
    em_rule = "l.inn = r.inn"
    inn_exact_levels = [
        lvl
        for comp in linker._settings_obj.comparisons
        if comp._output_column_name == "inn"
        for lvl in comp.comparison_levels
        if '"inn_l"' in lvl.sql_condition
        and '"inn_r"' in lvl.sql_condition
        and "=" in lvl.sql_condition
    ]
    if not inn_exact_levels:
        raise RuntimeError("Не найден точный уровень сравнения для ИНН.")
    linker.estimate_parameters_using_expectation_maximisation(
        blocking_rule=em_rule,
        comparison_levels_to_reverse_blocking_rule=inn_exact_levels,
    )

    # Предсказание
    # Порог можно задать через переменную окружения SPLINK_THRESHOLD
    env_thr = os.environ.get("SPLINK_THRESHOLD")
    thr_from_env = float(env_thr) if env_thr else None
    thr = match_threshold if match_threshold is not None else thr_from_env
    threshold = max(thr if thr is not None else 0.2, 1e-6)
    print(f"Предсказываем пары (threshold={threshold})...")
    predictions = linker.predict(threshold_match_probability=threshold)
    pred_df = predictions.as_pandas_dataframe()

    # Маппинг обратно к ID
    pred_df["orc_id"] = pred_df["unique_id_l"].map(orc_id_map)
    pred_df["employee_id"] = pred_df["unique_id_r"].map(emp_id_map)

    base_result = pred_df.dropna(subset=["orc_id", "employee_id"])[
        ["orc_id", "employee_id", "match_probability"]
    ]

    # Fallback: если Splink не дал ни одной пары, используем точные совпадения по ИНН
    if base_result.empty:
        print("Splink вернул 0 пар. Строим fallback по точному ИНН...")
        fallback = (
            orcs_df[["row_id", "inn"]]
            .dropna(subset=["inn"])
            .merge(
                employees_df[["row_id", "inn"]].dropna(subset=["inn"]),
                on="inn",
                how="inner",
                suffixes=("_orc", "_emp"),
            )
        )
        base_result = fallback.rename(
            columns={"row_id_orc": "orc_id", "row_id_emp": "employee_id"}
        )
        base_result["match_probability"] = 0.5
        print(f"Fallback пар по ИНН: {len(base_result)}")
    result = (
        base_result.sort_values("match_probability", ascending=False)
        .reset_index(drop=True)
    )

    out_path = run_dir / "suspected_orcs.csv"
    result.to_csv(out_path, index=False)
    print(f"Готово. Результат: {out_path} (rows={len(result)})")

    # Сабмит: top-1 для каждого орка, даже если вероятность низкая
    top1 = (
        base_result.sort_values("match_probability", ascending=False)
        .groupby("orc_id", as_index=False)
        .head(1)
    )
    submit = (
        top1.rename(columns={"orc_id": "id", "employee_id": "orig_index"})
        [["id", "orig_index"]]
        .reset_index(drop=True)
    )
    submit_path = run_dir / "submission.parquet"
    submit.to_parquet(submit_path, index=False)
    print(f"Сабмит сохранен: {submit_path} (rows={len(submit)})")

    # Файл с настраиваемыми параметрами
    tunable = run_dir / "tunable_params.txt"
    tunable.write_text(
        "\n".join(
            [
                "Настраиваемые параметры:",
                "- Пороги jaro_winkler для ФИО: [0.99, 0.95, 0.88]",
                "- Правила блокировки (blocking_rules_to_generate_predictions): dob, inn, surname_std, prefix4, soundex",
                "- Правила EM (estimate_parameters_using_expectation_maximisation): dob, inn",
                "- Порог match_probability в predict(): 0.8",
                "- max_pairs в estimate_u_using_random_sampling: 1_000_000",
                "- EMPLOYEE_ID_OFFSET (сдвиг уникальных ID)",
                "- Пути к данным ORCS_PATH/EMPLOYEES_PATH и рабочая БД DuckDB",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

