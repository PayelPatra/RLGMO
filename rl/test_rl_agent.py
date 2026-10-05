# rl/test_rl_agent.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import DQN
import torch
import pandas as pd
import pickle
import ast
import re

from env.medical_env import MedicalRecommendationEnv, tokenize_terms

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EMBED_DIR = ROOT / "embeddings"
MODEL_DIR = ROOT / "models"
OUT_FILE = ROOT / "rl" / "rl_test_results.csv"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def to_list_safe(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            v = ast.literal_eval(s)
            return v if isinstance(v, list) else []
        except Exception:
            return []
    return [t.strip() for t in re.split(r"[;,]\s*", s) if t.strip()]


# Load model + artifacts
model = DQN.load(str(MODEL_DIR / "trained_rl_model"))

gnn_embeddings = torch.load(EMBED_DIR / "gnn_embeddings.pt", map_location="cpu")
with open(EMBED_DIR / "node_mapping.pkl", "rb") as f:
    node_map = pickle.load(f)

with open(DATA_DIR / "med_list.pkl", "rb") as f:
    med_list = pickle.load(f)

# Load train + test
train_df = pd.read_csv(DATA_DIR / "extracted_annotations_cleaned.csv", dtype=str)
test_df = pd.read_csv(DATA_DIR / "test_cleaned.csv", dtype=str)

for df in (train_df, test_df):
    df.columns = df.columns.str.strip()
    df["RLGMO_ID"] = df["RLGMO_ID"].astype(str)
    for col in ["Symptoms", "Disease", "Medication"]:
        df[col] = df[col].apply(to_list_safe)
    df["Medication"] = df["Medication"].apply(lambda xs: [str(m).strip().lower() for m in xs if str(m).strip()])

train_records = train_df.set_index("RLGMO_ID")[["Symptoms", "Disease", "Medication"]].T.to_dict()
test_records = test_df.set_index("RLGMO_ID")[["Symptoms", "Disease", "Medication"]].T.to_dict()

# reference rules
reference_df = pd.read_csv(DATA_DIR / "Patient_data.csv", dtype=str)
reference_df["diagnose"] = reference_df["diagnose"].apply(lambda x: tokenize_terms(str(x)) if pd.notna(x) else set())
reference_df["pid"] = reference_df["pid"].apply(lambda x: str(x).strip().lower() if pd.notna(x) else "")
reference_df["did"] = reference_df["did"].apply(lambda x: str(x).strip() if pd.notna(x) else "")
reference_guidelines = [
    (row["diagnose"], [row["pid"]], row["did"])
    for _, row in reference_df.iterrows()
    if row["diagnose"] and row["pid"]
]

confidence = {rid: {m: 0.0 for m in info["Medication"]} for rid, info in {**train_records, **test_records}.items()}

env = MedicalRecommendationEnv(
    patient_records={**train_records, **test_records},
    med_list=med_list,
    gnn_embeddings=gnn_embeddings,
    node_map=node_map,
    reference_rules=reference_guidelines,
    gnn_confidence_scores=confidence,
    full_population=train_records,
)

results = []
present_ids = [rid for rid in test_df["RLGMO_ID"].astype(str) if rid in node_map]

if not present_ids:
    print("No results produced.")
    print("This means NONE of your test patient IDs exist in embeddings/node_mapping.")
    print(f"Test IDs: {len(test_df)} | Present in node_map: 0")
    print("Fix: include test patients when building the knowledge graph, then retrain GCN embeddings.")
    pd.DataFrame([]).to_csv(OUT_FILE, index=False)
    raise SystemExit(0)

for rid in present_ids:
    env.current_rlgmo_id = rid
    idx = node_map[rid]
    obs = gnn_embeddings[idx].detach().cpu().numpy().astype("float32").reshape(1, -1)

    action, _ = model.predict(obs, deterministic=True)
    aidx = int(action[0]) if hasattr(action, "__len__") else int(action)
    _, reward, _, _, info = env.step(aidx)

    results.append({
        "RLGMO_ID": rid,
        "selected_medication": info.get("selected_medication"),
        "actual_medication": info.get("actual_medication"),
        "reward": reward,
        "match_type": info.get("match_type"),
        "matched_terms": info.get("matched_terms"),
        "cosine_similarity": info.get("cosine_similarity"),
        "matched_RLGMO_ID": info.get("matched_RLGMO_ID"),
        "matched_diagnosis_terms": info.get("matched_diagnosis_terms"),
        "reference_id": info.get("reference_id"),
        "reference_treatment": info.get("reference_treatment"),
    })

df_results = pd.DataFrame(results)
df_results.to_csv(OUT_FILE, index=False)
print(f"RL test results saved to: {OUT_FILE}")
print(df_results.head(10))
