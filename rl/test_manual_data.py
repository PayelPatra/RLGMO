import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import torch
import pandas as pd
import pickle
import ast

from env.medical_env import MedicalRecommendationEnv, tokenize_terms

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EMBED_DIR = ROOT / "embeddings"
MODEL_DIR = ROOT / "models"
OUT_PATH = ROOT / "rl" / "rl_test_manual_results.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


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


def pick_id_col(df: pd.DataFrame) -> str:
    for c in ["RLGMO_ID", "Patient_ID", "patient_id", "pid"]:
        if c in df.columns:
            return c
    raise KeyError(f"No patient id column found. Columns: {list(df.columns)}")


def unpack_step(step_result):
    # Gym: (obs, reward, done, info)
    # Gymnasium: (obs, reward, terminated, truncated, info)
    if len(step_result) == 4:
        obs, reward, done, info = step_result
        return obs, reward, bool(done), info
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        done = bool(terminated) or bool(truncated)
        return obs, reward, done, info
    raise ValueError(f"Unexpected env.step() return length: {len(step_result)}")


# === 1) Manual test patients ===
test_data = [
    {"Patient_ID": "P41", "Symptoms": ["fever", "chills", "headache"], "Disease": ["malaria"], "Medication": ["chloroquine"]},
    {"Patient_ID": "P42", "Symptoms": ["cough", "sore throat", "fatigue"], "Disease": ["influenza"], "Medication": ["oseltamivir"]},
    {"Patient_ID": "P43", "Symptoms": ["joint pain", "rash", "nausea"], "Disease": ["dengue"], "Medication": ["paracetamol"]},
    {"Patient_ID": "P44", "Symptoms": ["chest pain", "shortness of breath", "sweating"], "Disease": ["angina"], "Medication": ["nitroglycerin"]},
    {"Patient_ID": "P45", "Symptoms": ["weight loss", "night sweats", "cough"], "Disease": ["tuberculosis"], "Medication": ["rifampicin"]},
    {"Patient_ID": "P46", "Symptoms": ["headache", "stiff neck", "fever"], "Disease": ["meningitis"], "Medication": ["ceftriaxone"]},
    {"Patient_ID": "P47", "Symptoms": ["vomiting", "diarrhea", "abdominal pain"], "Disease": ["gastroenteritis"], "Medication": ["ondansetron"]},
    {"Patient_ID": "P48", "Symptoms": ["blurred vision", "eye pain", "redness"], "Disease": ["conjunctivitis"], "Medication": ["ciprofloxacin"]},
    {"Patient_ID": "P49", "Symptoms": ["back pain", "fatigue", "numbness"], "Disease": ["sciatica"], "Medication": ["ibuprofen"]},
    {"Patient_ID": "P50", "Symptoms": ["insomnia", "anxiety", "palpitations"], "Disease": ["panic disorder"], "Medication": ["alprazolam"]},
    {"Patient_ID": "P51", "Symptoms": ["frequent urination", "thirst", "fatigue"], "Disease": ["diabetes"], "Medication": ["metformin"]},
    {"Patient_ID": "P52", "Symptoms": ["jaundice", "nausea", "abdominal pain"], "Disease": ["hepatitis"], "Medication": ["interferon"]},
    {"Patient_ID": "P53", "Symptoms": ["dry cough", "fever", "breathlessness"], "Disease": ["covid-19"], "Medication": ["remdesivir"]},
    {"Patient_ID": "P54", "Symptoms": ["rash", "joint pain", "conjunctivitis"], "Disease": ["zika"], "Medication": ["acetaminophen"]},
    {"Patient_ID": "P55", "Symptoms": ["swollen lymph nodes", "fever", "fatigue"], "Disease": ["mononucleosis"], "Medication": ["acyclovir"]},
]

test_df = pd.DataFrame(test_data)

# Normalize manual data
for col in ["Symptoms", "Disease", "Medication"]:
    test_df[col] = test_df[col].apply(lambda xs: [str(i).strip().lower() for i in xs])

test_records = test_df.set_index("Patient_ID")[["Symptoms", "Disease", "Medication"]].T.to_dict()
test_node_mapping = {pid: i for i, pid in enumerate(test_df["Patient_ID"])}

# === 2) Manual embeddings (same dimension as trained GNN embeddings) ===
gnn_embeddings = torch.load(EMBED_DIR / "gnn_embeddings.pt", map_location="cpu")
embedding_dim = int(gnn_embeddings.shape[1])
test_gnn_embeddings = torch.randn(len(test_df), embedding_dim)

test_confidence = {
    pid: {med: 0.0 for med in info["Medication"]}
    for pid, info in test_records.items()
}

# === 3) Load training population (for similarity matching / reference population) ===
train_df = pd.read_csv(DATA_DIR / "extracted_annotations_cleaned.csv", dtype=str)

train_id_col = pick_id_col(train_df)
train_df[train_id_col] = train_df[train_id_col].astype(str)

for col in ["Symptoms", "Disease", "Medication"]:
    if col in train_df.columns:
        train_df[col] = train_df[col].apply(parse_list_cell)
    else:
        train_df[col] = [[] for _ in range(len(train_df))]

train_df["Medication"] = train_df["Medication"].apply(
    lambda meds: [str(m).strip().lower() for m in meds if str(m).strip()]
)

train_records = (
    train_df.set_index(train_id_col)[["Symptoms", "Disease", "Medication"]]
    .T.to_dict()
)

# === 4) Reference rules ===
reference_df = pd.read_csv(DATA_DIR / "Patient_data.csv", dtype=str)
reference_df["diagnose"] = reference_df["diagnose"].apply(
    lambda x: tokenize_terms(str(x)) if pd.notna(x) and str(x).strip() else set()
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

# === 5) Medication list ===
with open(DATA_DIR / "med_list.pkl", "rb") as f:
    all_medications = pickle.load(f)

# === 6) Env ===
test_env = Monitor(
    MedicalRecommendationEnv(
        patient_records=test_records,
        med_list=all_medications,
        gnn_embeddings=test_gnn_embeddings,
        node_map=test_node_mapping,
        reference_rules=reference_guidelines,
        gnn_confidence_scores=test_confidence,
        full_population=train_records,
    )
)

# === 7) Load model ===
model = DQN.load(str(MODEL_DIR / "trained_rl_model"))

# === 8) Evaluate (one episode) ===
reset_result = test_env.reset()
obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

done = False
results = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = unpack_step(test_env.step(action))

    results.append({
        "patient_id": info.get("patient_id"),
        "selected_medication": info.get("selected_medication"),
        "actual_medication": info.get("actual_medication"),
        "matched_patient_id": info.get("matched_patient_id"),
        "similarity_score": info.get("similarity_score"),
        "reward": reward,
        "matched_terms": info.get("matched_terms"),
        "reference_match": info.get("reference_match"),
        "matched_diagnosis_terms": info.get("matched_diagnosis_terms"),
        "reference_id": info.get("reference_id"),
        "reference_treatment": info.get("reference_treatment"),
        "cosine_similarity": info.get("cosine_similarity"),
    })

df_result = pd.DataFrame(results)
df_result.to_csv(OUT_PATH, index=False)
print(f"Manual test results saved to: {OUT_PATH}")
print(df_result.head(10))
