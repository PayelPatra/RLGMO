from pathlib import Path
import pandas as pd
import ast
import re

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

TRAIN_CLEAN = DATA_DIR / "extracted_annotations_cleaned.csv"
TEST_RAW = DATA_DIR / "extracted_annotations_test.csv"
OUT_FILE = DATA_DIR / "test_cleaned.csv"


def clean_list_entry(entry):
    if isinstance(entry, list):
        return entry
    if pd.isna(entry):
        return []
    s = str(entry).strip()
    try:
        if s.startswith("[") and s.endswith("]"):
            v = ast.literal_eval(s)
            return v if isinstance(v, list) else []
    except Exception:
        pass
    return [t.strip() for t in re.split(r",|\n|;", s) if t.strip()]


def main():
    if not TRAIN_CLEAN.exists():
        raise FileNotFoundError(f"Missing cleaned train file: {TRAIN_CLEAN}")
    if not TEST_RAW.exists():
        raise FileNotFoundError(f"Missing raw test file: {TEST_RAW}")

    train_df = pd.read_csv(TRAIN_CLEAN, dtype=str)
    test_df = pd.read_csv(TEST_RAW, dtype=str)

    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()

    start_idx = len(train_df) + 1
    if "RLGMO_ID" not in test_df.columns:
        test_df["RLGMO_ID"] = [
            f"XYZ{str(i).zfill(3)}"
            for i in range(start_idx, start_idx + len(test_df))
        ]

    for col, default in [("Age", 0), ("Gender", -1), ("Cancer stage", -1)]:
        if col in test_df.columns:
            test_df[col] = pd.to_numeric(test_df[col], errors="coerce").fillna(default)

    for col in ["Symptoms", "Disease", "Medication", "Dose"]:
        if col in test_df.columns:
            test_df[col] = test_df[col].apply(clean_list_entry)

    test_df.to_csv(OUT_FILE, index=False)
    print(f" Saved cleaned test data: {OUT_FILE}")


if __name__ == "__main__":
    main()
