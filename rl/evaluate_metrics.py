# rl/evaluate_metrics.py
from pathlib import Path
import pandas as pd
import ast

ROOT = Path(__file__).resolve().parents[1]
RL_DIR = ROOT / "rl"
DATA_DIR = ROOT / "data"

PRED_FILE = RL_DIR / "rl_test_results.csv"
TEST_FILE = DATA_DIR / "test_cleaned.csv"
OUT_FILE = RL_DIR / "rl_test_with_eval.csv"


def parse_list_cell(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    s = str(x).strip()
    if s == "":
        return []
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, list) else []
    except Exception:
        return [t.strip() for t in s.split(",") if t.strip()]


if not PRED_FILE.exists():
    raise FileNotFoundError(f"Predictions file not found: {PRED_FILE}")

# handle empty file safely
if PRED_FILE.stat().st_size == 0:
    print("Predictions file is empty -> skipping evaluation.")
    pd.DataFrame([]).to_csv(OUT_FILE, index=False)
    raise SystemExit(0)

pred_df = pd.read_csv(PRED_FILE, dtype=str)

if pred_df.empty or pred_df.columns.size == 0:
    print("Predictions dataframe empty -> skipping evaluation.")
    pd.DataFrame([]).to_csv(OUT_FILE, index=False)
    raise SystemExit(0)

# unify id column
if "patient_id" in pred_df.columns and "RLGMO_ID" not in pred_df.columns:
    pred_df = pred_df.rename(columns={"patient_id": "RLGMO_ID"})

if "RLGMO_ID" not in pred_df.columns:
    raise KeyError(f"RLGMO_ID missing in predictions columns: {list(pred_df.columns)}")

pred_df["RLGMO_ID"] = pred_df["RLGMO_ID"].astype(str)
pred_df["selected_medication"] = pred_df["selected_medication"].astype(str).str.strip().str.lower()

df_test = pd.read_csv(TEST_FILE, dtype=str)
df_test["RLGMO_ID"] = df_test["RLGMO_ID"].astype(str)
df_test["Medication"] = df_test["Medication"].apply(parse_list_cell)
df_test["Medication"] = df_test["Medication"].apply(
    lambda meds: [str(m).strip().lower() for m in meds if str(m).strip()]
)

merged = pred_df.merge(df_test[["RLGMO_ID", "Medication"]], on="RLGMO_ID", how="left")
merged["Medication"] = merged["Medication"].apply(lambda x: x if isinstance(x, list) else [])

merged["Match"] = [
    int(sel in set(meds))
    for sel, meds in zip(merged["selected_medication"], merged["Medication"])
]

tp = int(merged["Match"].sum())
fp = int(len(merged) - tp)
fn = int(sum(1 for _, r in merged.iterrows() if r["Medication"] and r["selected_medication"] not in r["Medication"]))

precision = tp / (tp + fp) if (tp + fp) else 0.0
recall = tp / (tp + fn) if (tp + fn) else 0.0
f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
accuracy = tp / len(merged) if len(merged) else 0.0

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

merged.to_csv(OUT_FILE, index=False)
print(f"Evaluation results saved to: {OUT_FILE}")
