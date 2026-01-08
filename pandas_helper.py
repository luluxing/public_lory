import pandas as pd
from pathlib import Path
import helpers
from custom_lib import parallel

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
    Returns {table_name: dataframe}
    """
    all_tables_dir = lake_dir / "all_tables"
    tables: dict[str, pd.DataFrame] = {}

    csv_paths = helpers.list_csv_files(all_tables_dir)  # assumed to return Iterable[Path]
    argument_list = [{"path": str(p), "name": p.stem} for p in csv_paths]

    # If your executor supports worker config, pass via common_arguments
    results = parallel.execute(
        func=_read_csv_worker,
        argument_list=argument_list
    )

    for status, name, payload in results:
        if status == "ok":
            tables[name] = payload  # payload is the DataFrame
        else:
            # payload is the error message
            helpers.eprint(f"⚠️  Skipping {name}: {payload}")
    return tables