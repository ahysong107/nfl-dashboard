"""Fetch the injuries release. Like depth_charts, this grows through the
week (Wed/Thu/Fri practice reports, then a final game-status designation),
so it's refetched fresh every run rather than cached once.
"""
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    url = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_2026.parquet"
    inj = pd.read_parquet(url)
    path = os.path.join(DATA_DIR, "injuries_2026.parquet")
    inj.to_parquet(path)
    print(f"saved {path}, {inj.shape}")
    if len(inj):
        print("weeks with a report filed:", sorted(inj["week"].unique()))
        print(inj["report_status"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
