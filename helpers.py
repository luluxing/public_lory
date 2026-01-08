import pandas as pd
from pathlib import Path
import sys
import codecs
def print_df(df: pd.DataFrame, max_rows: int = 30):
    if len(df) > max_rows:
        print(df.head(max_rows).to_string(index=True))
        print(f"... ({len(df) - max_rows} more rows)")
    else:
        print(df.to_string(index=True))

def remove_non_utf8_lines(filename):
    with codecs.open(filename, 'r', encoding='utf-8', errors='ignore') as f_in:
        lines = f_in.readlines()

    with codecs.open(filename, 'w', encoding='utf-8') as f_out:
        for line in lines:
            try:
                line.encode('utf-8')
                f_out.write(line)
            except UnicodeEncodeError:
                continue

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)



def list_dir_names(d: Path) -> list[str]:
    return sorted([p.name for p in d.iterdir() if p.is_dir()]) if d.exists() else []

def list_csv_files(d: Path) -> list[Path]:
    return sorted(d.glob("*.csv")) if d.exists() else []

def choose_from_list(prompt_text: str, options: list[str]) -> str:
    if not options:
        raise RuntimeError("No options to choose from.")
    print(f"\n{prompt_text}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        choice = input("Enter choice number: ").strip()
        if not choice.isdigit():
            print("Please enter a valid number.")
            continue
        idx = int(choice)
        if 1 <= idx <= len(options):
            return options[idx - 1]
        print("Out of range. Try again.")

def print_rule(title: str = ""):
    line = "─" * 80
    if title:
        print(f"\n{line}\n{title}\n{line}")
    else:
        print(f"\n{line}")

def yes_no(prompt_text: str, default_yes: bool = True) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    while True:
        ans = input(f"{prompt_text} {suffix} ").strip().lower()
        if ans == "" and default_yes:
            return True
        if ans == "" and not default_yes:
            return False
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please answer 'y' or 'n'.")




def export_df(df: pd.DataFrame, out_path: Path | None):
    if out_path is None:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✅ Saved to {out_path}")

def split_dict(original_dict, n):
    """
    Split a dictionary into a list of dictionaries,
    each containing up to n key-value pairs.

    :param original_dict: dict, the dictionary to split
    :param n: int, max number of items in each split dictionary
    :return: list of dictionaries
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")

    items = list(original_dict.items())
    return [dict(items[i:i + n]) for i in range(0, len(items), n)]

def split_list(original_list, n):
    """
    Split a list into a list of lists,
    each containing up to n items.
    :param original_list: list, the list to split
    :param n: int, max number of items in each split list
    :return: list of lists
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    return [original_list[i:i + n] for i in range(0, len(original_list), n)]