"""Utility to compare class samples between employees and orcs."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


def load_samples(report_dir: Path) -> dict[str, list[dict]]:
    path = report_dir / "class_samples.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def side_by_side(emp_dir="reports/name/employees", orcs_dir="reports/name/orcs", classes=None):
    if classes is None:
        classes = ["corrected", "untouched", "ambiguous", "empty", "no_match"]
    emp = load_samples(Path(emp_dir))
    orc = load_samples(Path(orcs_dir))
    result = {}
    for cls in classes:
        emp_df = pd.DataFrame(emp.get(cls, [])).head(20)
        orc_df = pd.DataFrame(orc.get(cls, [])).head(20)
        emp_df = emp_df.add_suffix("_employees")
        orc_df = orc_df.add_suffix("_orcs")
        result[cls] = pd.concat(
            [emp_df.reset_index(drop=True), orc_df.reset_index(drop=True)],
            axis=1,
        )
    return result


if __name__ == "__main__":
    tables = side_by_side()
    for cls, df in tables.items():
        print(f"\n=== {cls} ===")
        print(df.head(20))








