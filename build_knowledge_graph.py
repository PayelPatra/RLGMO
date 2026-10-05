# graph/build_knowledge_graph.py
from __future__ import annotations

from pathlib import Path
import pickle
import ast
import re
import pandas as pd
import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
GRAPH_DIR = ROOT / "graph"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "extracted_annotations_cleaned.csv"
TEST_FILE = DATA_DIR / "test_cleaned.csv"
REF_FILE = DATA_DIR / "Patient_data.csv"

OUT_GRAPH = GRAPH_DIR / "knowledge_graph.pkl"


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


def tokenize_terms(text: str):
    return set(re.findall(r"\b[a-zA-Z]+\b", str(text).lower()))


def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Missing: {TRAIN_FILE}")
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Missing: {TEST_FILE}")
    if not REF_FILE.exists():
        raise FileNotFoundError(f"Missing: {REF_FILE}")

    train_df = pd.read_csv(TRAIN_FILE, dtype=str)
    test_df = pd.read_csv(TEST_FILE, dtype=str)

    # Ensure id column name consistency
    for df in (train_df, test_df):
        df.columns = df.columns.str.strip()
        if "RLGMO_ID" not in df.columns:
            raise KeyError(f"RLGMO_ID missing in {df}")

    # Combine train + test so KG contains BOTH
    df_all = pd.concat([train_df, test_df], ignore_index=True)

    # Parse list columns
    for col in ["Symptoms", "Disease", "Medication", "Dose"]:
        if col in df_all.columns:
            df_all[col] = df_all[col].apply(to_list_safe)
        else:
            df_all[col] = [[] for _ in range(len(df_all))]

    G = nx.Graph()

    # Add patient nodes and edges
    for _, row in df_all.iterrows():
        rid = str(row["RLGMO_ID"]).strip()
        if not rid:
            continue

        G.add_node(rid, type="patient")

        for s in row.get("Symptoms", []):
            s = str(s).strip().lower()
            if s:
                G.add_node(s, type="symptom")
                G.add_edge(rid, s)

        for d in row.get("Disease", []):
            d = str(d).strip().lower()
            if d:
                G.add_node(d, type="disease")
                G.add_edge(rid, d)

        for m in row.get("Medication", []):
            m = str(m).strip().lower()
            if m:
                G.add_node(m, type="medication")
                G.add_edge(rid, m)

        for dose in row.get("Dose", []):
            dose = str(dose).strip().lower()
            if dose:
                G.add_node(dose, type="dose")
                G.add_edge(rid, dose)

        stage = row.get("Cancer stage", None)
        if pd.notna(stage):
            try:
                stage_node = f"stage_{int(float(stage))}"
                G.add_node(stage_node, type="stage")
                G.add_edge(rid, stage_node)
            except Exception:
                pass

    # Add reference guideline nodes
    reference_df = pd.read_csv(REF_FILE, dtype=str)
    reference_df.columns = reference_df.columns.str.strip()

    if "diagnose" in reference_df.columns and "pid" in reference_df.columns and "did" in reference_df.columns:
        reference_df["diagnose"] = reference_df["diagnose"].apply(
            lambda x: tokenize_terms(x) if pd.notna(x) else set()
        )
        reference_df["pid"] = reference_df["pid"].apply(
            lambda x: str(x).strip().lower() if pd.notna(x) else ""
        )
        reference_df["did"] = reference_df["did"].apply(
            lambda x: str(x).strip() if pd.notna(x) else ""
        )

        for _, row in reference_df.iterrows():
            diag_terms = row["diagnose"]
            treat = row["pid"]
            ref_id = row["did"]

            if not diag_terms or not treat:
                continue

            ref_node = f"REF_{ref_id}"
            G.add_node(ref_node, type="reference")

            for term in diag_terms:
                term = str(term).strip().lower()
                if term:
                    G.add_node(term, type="diagnose")
                    G.add_edge(ref_node, term)

            G.add_node(treat, type="treatment")
            G.add_edge(ref_node, treat)

    with open(OUT_GRAPH, "wb") as f:
        pickle.dump(G, f)

    print(f"Knowledge Graph saved: {OUT_GRAPH}")
    print(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")


if __name__ == "__main__":
    main()
