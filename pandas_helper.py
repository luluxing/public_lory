from __future__ import annotations

from pathlib import Path

import pandas as pd

import helpers
from custom_lib import parallel


def temp_tables_dir() -> Path:
    path = Path(__file__).resolve().parent / "temp_tables"
    path.mkdir(parents=True, exist_ok=True)
    return path


def temp_table_path(table_name: str) -> Path:
    return temp_tables_dir() / f"{table_name}.csv"


def clear_temp_table_csvs(temp_dir: Path | None = None) -> list[Path]:
    """Delete persisted temp-table artifacts and return the removed paths."""
    tables_dir = temp_dir or temp_tables_dir()
    if not tables_dir.exists():
        return []

    removed_paths: list[Path] = []
    for csv_path in sorted(tables_dir.iterdir()):
        if not csv_path.is_file():
            continue
        if (
            csv_path.suffix.lower() != ".csv"
            and not csv_path.name.endswith(".geo_association.json")
        ):
            continue
        csv_path.unlink(missing_ok=True)
        removed_paths.append(csv_path)
    return removed_paths


def read_csv_safely(path: Path) -> pd.DataFrame:
    # remove_non_utf8_lines(path)
    return pd.read_csv(path, on_bad_lines='skip', dtype=str)

# ───────────────────────── worker for one CSV ─────────────────────────
def _read_csv_worker(arg, common_arguments, common_args_for_batch):
    """
    arg: {"path": "<string path>", "name": "<table_name>"}
    returns:
      ("ok", name, df) on success
      ("err", name, "message") on failure
    """
    p = Path(arg["path"])
    name = arg["name"]

    df = read_csv_safely(p)
    return ("ok", name, df)


# ───────────────────── parallelized table loader ─────────────────────
def load_datalake_tables(lake_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Reads all CSVs under lake_dir / 'all_tables' into DataFrames in parallel.
    If temp_tables/<table>.csv exists, it overrides the base lake copy for that table.
    Returns {table_name: dataframe}.
    """
    all_tables_dir = lake_dir / "all_tables"
    tables: dict[str, pd.DataFrame] = {}

    csv_paths = helpers.list_csv_files(all_tables_dir)  # assumed to return Iterable[Path]
    argument_list = [{"path": str(p), "name": p.stem} for p in csv_paths]

    # If your executor supports worker config, pass via common_arguments
    results = parallel.execute(
        func=_read_csv_worker,
        argument_list=argument_list,
        # Streamlit on macOS can fail when the custom multiprocessing wrapper
        # spins up a Manager; table loading is small enough to keep serial.
        parallel_enabled=False,
    )

    for status, name, payload in results:
        if status == "ok":
            tables[name] = payload  # payload is the DataFrame
        else:
            # payload is the error message
            helpers.eprint(f"⚠️  Skipping {name}: {payload}")

    _overlay_temp_table_csvs(tables)
    return tables


def _normalize_temp_table_schema(
    base_df: pd.DataFrame,
    temp_df: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    """
    Ensure a temp table remains "original columns plus extra geo columns".

    Older temp tables dropped some source columns during geocoding. When the temp
    copy still matches the base row count, restore any missing base columns and keep
    temp-table columns appended.
    """
    missing_base_cols = [col for col in base_df.columns if col not in temp_df.columns]
    if not missing_base_cols or len(base_df) != len(temp_df):
        return temp_df, False

    normalized_df = base_df.reset_index(drop=True).copy()
    temp_reset = temp_df.reset_index(drop=True)
    for col in temp_reset.columns:
        normalized_df[col] = temp_reset[col]
    return normalized_df, True


def _overlay_temp_table_csvs(tables: dict[str, pd.DataFrame]) -> None:
    """
    Prefer authoritative temp-table copies over raw all_tables CSVs.

    Each temp-table file is stored as temp_tables/<table>.csv.
    When a temp-table exists for a base lake table, it replaces that table in memory.
    """
    tables_dir = temp_tables_dir()
    if not tables_dir.is_dir():
        return

    for path in sorted(tables_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".csv":
            continue
        table_name = path.stem
        if table_name not in tables:
            continue
        try:
            temp_df = read_csv_safely(path)
            normalized_df, was_repaired = _normalize_temp_table_schema(
                base_df=tables[table_name],
                temp_df=temp_df,
            )
            if was_repaired:
                normalized_df.to_csv(path, index=False)
            tables[table_name] = normalized_df
        except Exception as exc:
            helpers.eprint(f"⚠️  Skipping temp table {table_name}: {exc}")
