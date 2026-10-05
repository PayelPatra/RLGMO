# RLGMO: Reinforcement Learning Guided Medication Optimization

## Overview

This repository contains the implementation of a Reinforcement Learning Guided Medication Optimization (RLGMO) framework for personalized medication recommendation in cancer care.

The framework combines:

- Data preprocessing
- Medical knowledge graph construction
- Graph Convolutional Network (GCN)-based representation learning
- Patient and medication pattern analysis
- Reinforcement Learning (RL) for medication recommendation
- Recommendation evaluation
- Visualization of learned representations and recommendations

The overall workflow integrates patient information, cancer-related knowledge, medication information, and learned patient representations to support personalized medication recommendation.

## Framework Workflow

```text
Patient Data
     |
     v
Data Preprocessing
     |
     +-- Training Data Cleaning
     +-- Test Data Cleaning
     |
     v
Knowledge Graph Construction
     |
     v
GCN Representation Learning
     |
     v
Patient/Pattern Similarity Analysis
     |
     v
Reinforcement Learning Environment
     |
     v
DQN-Based Medication Recommendation
     |
     v
Testing and Evaluation
     |
     +-- Recommendation Evaluation
     +-- Medication Similarity
     +-- Visualization
```

## Repository Structure

```text
RLGMO/
|
+-- README.md
+-- requirements.txt
+-- main.py
|
+-- data/
|
+-- data_preprocessing/
|   +-- clean_train_data.py
|   +-- clean_test_data.py
|
+-- graph/
|   +-- build_knowledge_graph.py
|   +-- plot_gcn_patterns.py
|   +-- plot_patient_subgraph.py
|
+-- embeddings/
|   +-- cancer_stage_coloring.py
|   +-- pattern_similarity.py
|   +-- train_gcn.py
|   +-- visualize_embeddings.py
|   +-- visualize_patterns.py
|
+-- env/
|   +-- medical_env.py
|
+-- rl/
    +-- train_rl_agent.py
    +-- test_rl_agent.py
    +-- test_manual_data.py
    +-- evaluate_metrics.py
    +-- plot_med_recommendations.py
    +-- plot_med_similarity_score.py
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

Create and activate a Python environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python -m venv venv
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data

The framework requires patient/cancer-related data as input.

The data should be placed in the appropriate `data/` directory.

Because patient-related medical information may contain sensitive or confidential information, raw patient-level datasets should not be publicly uploaded unless they are appropriately anonymized and cleared for public distribution.

For reproducibility, researchers should provide either:

- An anonymized dataset
- A synthetic/example dataset
- Instructions describing how the dataset can be obtained

## Step-by-Step Execution

### 1. Training Data Preprocessing

```bash
python data_preprocessing/clean_train_data.py
```

Prepares and cleans the training data required for downstream processing.

### 2. Test Data Preprocessing

```bash
python data_preprocessing/clean_test_data.py
```

Prepares the test data for evaluating the trained recommendation model.

### 3. Knowledge Graph Construction

```bash
python graph/build_knowledge_graph.py
```

Constructs the medical knowledge graph from the processed data.

### 4. GCN Representation Learning

```bash
python embeddings/train_gcn.py
```

Trains the Graph Convolutional Network (GCN) and learns graph-based representations from the constructed knowledge graph.

### 5. Pattern Similarity Analysis

```bash
python embeddings/pattern_similarity.py
```

Analyzes similarity between learned patient patterns using the learned representations.

### 6. Embedding Visualization

```bash
python embeddings/visualize_embeddings.py
```

Visualizes the learned graph embeddings.

### 7. Cancer Stage Visualization

```bash
python embeddings/cancer_stage_coloring.py
```

Generates visualization based on cancer-stage information.

### 8. Pattern Visualization

```bash
python embeddings/visualize_patterns.py
```

Visualizes learned patient or graph patterns.

### 9. GCN Pattern Visualization

```bash
python graph/plot_gcn_patterns.py
```

Generates visualizations of GCN-derived patterns.

### 10. Patient Subgraph Visualization

```bash
python graph/plot_patient_subgraph.py
```

Visualizes patient-related subgraphs from the knowledge graph.

### 11. Reinforcement Learning Environment

The reinforcement learning environment is implemented in:

```text
env/medical_env.py
```

The environment defines the patient state, medication actions, and reward mechanism used by the reinforcement learning agent.

### 12. DQN Training

```bash
python rl/train_rl_agent.py
```

Trains the Deep Q-Network (DQN) agent for medication recommendation.

### 13. RL Testing

```bash
python rl/test_rl_agent.py
```

Tests the trained reinforcement learning agent and generates medication recommendations for test cases.

### 14. Manual Testing

```bash
python rl/test_manual_data.py
```

Allows manually specified patient data to be tested through the recommendation pipeline.

### 15. Evaluation

```bash
python rl/evaluate_metrics.py
```

Evaluates the medication recommendation performance.

### 16. Medication Recommendation Visualization

```bash
python rl/plot_med_recommendations.py
```

Generates plots of medication recommendation results.

### 17. Medication Similarity Visualization

```bash
python rl/plot_med_similarity_score.py
```

Generates plots of medication similarity scores.

### 18. Complete Pipeline

```bash
python main.py
```

Runs the overall project workflow where supported by the configured project structure.

## Reproducibility

For reproducibility, follow the processing order described below:

```text
1. Prepare the input data
2. Clean the training data
3. Clean the test data
4. Construct the medical knowledge graph
5. Train the GCN
6. Generate graph embeddings
7. Perform pattern similarity analysis
8. Prepare the reinforcement learning environment
9. Train the DQN agent
10. Test the trained agent
11. Evaluate recommendation performance
12. Generate visualizations
```

The required input data, intermediate files, trained models, and generated results should be maintained consistently between stages.

## Expected Outputs

Depending on the execution stage, the framework may generate:

- Cleaned datasets
- Medical knowledge graph files
- Graph/node embeddings
- Patient-pattern similarity results
- Trained GCN representations
- Trained reinforcement learning models
- Medication recommendations
- Evaluation results
- Similarity scores
- Graph visualizations
- Embedding visualizations
- Patient subgraph visualizations

## Research Framework

The RLGMO framework integrates graph-based representation learning with reinforcement learning.

### Medical Knowledge Graph

Represents relationships between patient-related entities, cancer characteristics, medications, and other relevant medical information.

### Graph Convolutional Network

Learns graph-based representations from the medical knowledge graph.

### Pattern Similarity

Uses learned representations to identify similar patient patterns.

### Reinforcement Learning

Uses a DQN-based agent to learn medication recommendation policies through interaction with the medical environment.

### Evaluation

The trained agent is evaluated using test cases and recommendation-related metrics.

## Data Privacy and Ethics

Any dataset containing patient information must be handled according to applicable ethical, privacy, and data-protection requirements.

Patient-identifiable or confidential information should not be uploaded to this public repository.

If the original dataset cannot be publicly distributed, users should provide an appropriate description of the dataset and the procedure required to obtain access.

## Research Use

This repository contains research code for the RLGMO framework.

The system is a research prototype and should not be used directly for clinical treatment or patient-care decisions without appropriate clinical validation, regulatory review, and expert oversight.

## Citation

If you use this repository or the RLGMO framework in your research, please cite the associated publication:

```text
Citation will be added after publication.
```

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

