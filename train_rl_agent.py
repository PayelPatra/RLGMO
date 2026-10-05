import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import DQN
import pandas as pd
import pickle
import torch
import ast

from env.medical_env import MedicalRecommendationEnv

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EMBED_DIR = ROOT / "embeddings"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


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


# === LOAD TRAIN DATA ===
train_df = pd.read_csv(DATA_DIR / "extracted_annotations_cleaned.csv", dtype=str)

# Pick correct ID column (most common: RLGMO_ID)
id_col = None
for c in ["RLGMO_ID", "Patient_ID", "patient_id", "pid"]:
    if c in train_df.columns:
        id_col = c
        break

if id_col is None:
    raise KeyError(f"No patient id column found in train CSV. Columns: {list(train_df.columns)}")

train_df[id_col] = train_df[id_col].astype(str)

for col in ["Symptoms", "Disease", "Medication"]:
    if col in train_df.columns:
        train_df[col] = train_df[col].apply(parse_list_cell)
    else:
        train_df[col] = [[] for _ in range(len(train_df))]

train_df["Medication"] = train_df["Medication"].apply(
    lambda meds: [str(m).strip().lower() for m in meds if str(m).strip()]
)

# patient_records expects keys to match node_mapping patient nodes (usually RLGMO_ID like XYZ001)
train_records = (
    train_df.set_index(id_col)[["Symptoms", "Disease", "Medication"]]
    .T.to_dict()
)

# === LOAD REFERENCE RULES ===
reference_df = pd.read_csv(DATA_DIR / "Patient_data.csv", dtype=str)
reference_df["diagnose"] = reference_df["diagnose"].apply(
    lambda x: {str(x).strip().lower()} if pd.notna(x) and str(x).strip() else set()
)
reference_df["pid"] = reference_df["pid"].apply(
    lambda x: str(x).strip().lower() if pd.notna(x) and str(x).strip() else ""
)
reference_df["did"] = reference_df["did"].apply(
    lambda x: str(x).strip() if pd.notna(x) and str(x).strip() else ""
)

reference_guidelines = [
    (row["diagnose"], [row["pid"]], row["did"])
    for _, row in reference_df.iterrows()
    if row["diagnose"] and row["pid"]
]

# === MED LIST ===
all_medications = sorted({m for meds in train_df["Medication"] for m in meds})
with open(DATA_DIR / "med_list.pkl", "wb") as f:
    pickle.dump(all_medications, f)

# === LOAD NODE MAPPING + EMBEDDINGS ===
with open(EMBED_DIR / "node_mapping.pkl", "rb") as f:
    node_mapping = pickle.load(f)

gnn_embeddings = torch.load(EMBED_DIR / "gnn_embeddings.pt", map_location="cpu")
print(f"Loaded embeddings: {gnn_embeddings.shape}")

# Optional sanity warning: how many patient IDs match node_mapping?
missing = [pid for pid in train_records.keys() if pid not in node_mapping]
if missing:
    print(f"⚠ Warning: {len(missing)} patient IDs not found in node_mapping. Example: {missing[:5]}")
    print("   This usually means you used Patient_ID instead of RLGMO_ID/XYZ IDs.")

# === CONFIDENCE SCORES ===
gnn_confidence_scores = {
    pid: {med: 0.0 for med in info["Medication"]}
    for pid, info in train_records.items()
}

# === CREATE ENV + TRAIN ===
train_env = MedicalRecommendationEnv(
    patient_records=train_records,
    med_list=all_medications,
    gnn_embeddings=gnn_embeddings,
    node_map=node_mapping,
    reference_rules=reference_guidelines,
    gnn_confidence_scores=gnn_confidence_scores,
    full_population=train_records,
)

model = DQN("MlpPolicy", train_env, verbose=1)
model.learn(total_timesteps=50000)
model.save(str(MODEL_DIR / "trained_rl_model"))

print("RL training complete and model saved.")
