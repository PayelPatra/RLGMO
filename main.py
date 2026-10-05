# main.py
from __future__ import annotations

import sys
from pathlib import Path
import subprocess
import os

ROOT = Path(__file__).resolve().parent


def run_script(rel_path: str) -> None:
    """Run a project script with the current interpreter; fail fast on errors."""
    script = ROOT / rel_path
    if not script.exists():
        raise FileNotFoundError(f"Missing script: {script}")

    print(f"\n Running: {rel_path}")

    env = os.environ.copy()
    # Ensure project root is importable (so `from env...` works everywhere)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env.get("PYTHONPATH", ""))

    subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
        check=True,
        env=env
    )


def main() -> None:
    # Step 1: Clean data
    print(" Cleaning training and test data...")
    run_script("data_preprocessing/clean_train_data.py")
    run_script("data_preprocessing/clean_test_data.py")

    # Step 2: Build Knowledge Graph
    print(" Building knowledge graph...")
    run_script("graph/build_knowledge_graph.py")

    # Step 3: Train GCN model + embeddings + pattern files + visualizations
    print(" Training GCN model and generating embeddings...")
    run_script("embeddings/train_gcn.py")
    run_script("embeddings/pattern_similarity.py")
    run_script("embeddings/visualize_embeddings.py")
    run_script("embeddings/cancer_stage_coloring.py")
    run_script("embeddings/visualize_patterns.py")

    # Step 4: Visualize patient subgraph (optional)
    print(" Visualizing patient subgraph...")
    run_script("graph/plot_patient_subgraph.py")

    # Step 5: Run RL training + tests
    print(" Running RL agent training...")
    run_script("rl/train_rl_agent.py")
    run_script("rl/test_rl_agent.py")
    run_script("rl/test_manual_data.py")

    # Step 6: Evaluate + plots
    print(" Evaluating and plotting metrics...")
    run_script("rl/evaluate_metrics.py")
    run_script("rl/plot_med_recommendations.py")
    run_script("rl/plot_med_similarity_score.py")

    print("\n Pipeline execution completed.")


if __name__ == "__main__":
    main()
