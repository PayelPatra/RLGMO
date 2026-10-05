from pathlib import Path
import pandas as pd
import re
import ast
import nltk
from nltk.corpus import stopwords

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

IN_FILE = DATA_DIR / "extracted_annotations.csv"
OUT_FILE = DATA_DIR / "extracted_annotations_cleaned.csv"

nltk.download("stopwords")
STOP_WORDS = set(stopwords.words("english"))


def clean_list_entry(entry):
    if isinstance(entry, list):
        items = entry
    else:
        s = "" if pd.isna(entry) else str(entry)
        try:
            items = ast.literal_eval(s) if s.strip().startswith("[") else re.split(r",|\n|;", s)
        except Exception:
            items = re.split(r",|\n|;", s)

    cleaned = []
    for item in items:
        t = str(item).strip().lower()
        t = re.sub(r"[^a-z0-9\s\-]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        if not t:
            continue
        if t in STOP_WORDS:
            continue
        cleaned.append(t)
    return cleaned


def remove_dates(text_list):
    if not isinstance(text_list, list):
        return []
    date_patterns = [
        r"^\d{1,2}/\d{1,2}/\d{2,4}$",
        r"^\d{1,2}-\d{1,2}-\d{2,4}$",
        r"^\d{4}-\d{1,2}-\d{1,2}$",
    ]
    out = []
    for item in text_list:
        if any(re.fullmatch(p, item) for p in date_patterns):
            continue
        out.append(item)
    return out


def main():
    if not IN_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {IN_FILE}")

    df = pd.read_csv(IN_FILE, dtype=str)
    df.columns = df.columns.str.strip()

    if "RLGMO_ID" not in df.columns:
        df["RLGMO_ID"] = [f"XYZ{str(i+1).zfill(3)}" for i in range(len(df))]

    if "Gender" in df.columns:
        df["Gender"] = (
            df["Gender"].astype(str).str.lower()
            .map({"male": 0, "female": 1})
            .fillna(-1).astype(int)
        )

    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(0).astype(int)

    for col in ["Symptoms", "Disease", "Medication", "Dose"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_list_entry).apply(remove_dates)

    if "Cancer stage" in df.columns:
        df["Cancer stage"] = df["Cancer stage"].astype(str).str.strip()

    df.to_csv(OUT_FILE, index=False)
    print(f" Saved cleaned training data: {OUT_FILE}")


if __name__ == "__main__":
    main()
