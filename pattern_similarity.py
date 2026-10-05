from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import ast

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EMBED_DIR = ROOT / "embeddings"

TRAIN_FILE = DATA_DIR / "extracted_annotations_cleaned.csv"
OUT_FILE = EMBED_DIR / "gcn_based_rlgmo_patterns.csv"


def to_list_safe(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, list) else []
    except Exception:
        return [t.strip() for t in s.split(",") if t.strip()]


def load_embeddings_and_map():
    emb_candidates = [EMBED_DIR / "gnn_embeddings.pt", EMBED_DIR / "patient_gnn_embeddings.pt"]
    map_candidates = [EMBED_DIR / "node_mapping.pkl", EMBED_DIR / "patient_node_map.pkl"]

    emb_path = next((p for p in emb_candidates if p.exists()), None)
    map_path = next((p for p in map_candidates if p.exists()), None)

    if emb_path is None:
        raise FileNotFoundError(f"Missing embeddings file. Tried: {emb_candidates}")
    if map_path is None:
        raise FileNotFoundError(f"Missing node map file. Tried: {map_candidates}")

    gnn_embeddings = torch.load(emb_path, map_location="cpu")
    with open(map_path, "rb") as f:
        node_mapping = pickle.load(f)

    return gnn_embeddings, node_mapping


def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Missing: {TRAIN_FILE}")

    gnn_embeddings, node_mapping = load_embeddings_and_map()

    rev_node_mapping = {v: k for k, v in node_mapping.items()}
    gnn = gnn_embeddings.detach().cpu().float()

    patient_ids, patient_vecs = [], []
    for idx, node_name in rev_node_mapping.items():
        s = str(node_name)
        if s.startswith("XYZ"):
            patient_ids.append(s)
            patient_vecs.append(gnn[idx].numpy())

    if not patient_vecs:
        raise ValueError("No patient nodes found (expected IDs like XYZ001...)")

    patient_vecs = np.vstack(patient_vecs)
    sim = cosine_similarity(patient_vecs, patient_vecs)
    np.fill_diagonal(sim, -1.0)

    df = pd.read_csv(TRAIN_FILE, dtype=str)
    df["RLGMO_ID"] = df["RLGMO_ID"].astype(str)
    df["Disease"] = df["Disease"].apply(to_list_safe)
    df["Medication"] = df["Medication"].apply(to_list_safe)

    pid_to_disease = dict(zip(df["RLGMO_ID"], df["Disease"]))
    pid_to_med = dict(zip(df["RLGMO_ID"], df["Medication"]))

    rows = []
    for i, rid in enumerate(patient_ids):
        j = int(np.argmax(sim[i]))
        neighbor_rid = patient_ids[j]
        neighbor_diseases = [str(x).strip().lower() for x in pid_to_disease.get(neighbor_rid, []) if str(x).strip()]
        neighbor_meds = [str(x).strip().lower() for x in pid_to_med.get(neighbor_rid, []) if str(x).strip()]

        top_disease = neighbor_diseases[0] if neighbor_diseases else ""
        top_med = neighbor_meds[0] if neighbor_meds else ""

        rows.append({
            "RLGMO_ID": rid,
            "Nearest_RLGMO_ID": neighbor_rid,
            "CosineSimilarity": float(sim[i, j]),
            "Top_Disease": top_disease,
            "Top_Medication": top_med,
            "Pattern": f"{rid} → {neighbor_rid} → {top_disease} → {top_med}"
        })

    pattern_df = pd.DataFrame(rows)
    pattern_df.to_csv(OUT_FILE, index=False)
    print(f"Saved: {OUT_FILE}")
    print(pattern_df.head(10))


if __name__ == "__main__":
    main()
