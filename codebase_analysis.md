# Codebase Analysis Report: Candidate Ranking System

## 1. Overview
This codebase is an AI/ML-powered candidate ranking and retrieval system, seemingly built for a hackathon or challenge (referenced as `India_runs_data_and_ai_challenge`). It is designed to evaluate candidates against a specific Job Description (JD) using a combination of semantic similarity, rule-based heuristics, behavioral signals, and "honeypot" checks (to catch resume inflation/fake profiles).

The system is split into two phases:
- **Offline Phase (`precompute.py`)**: Heavy processing, embedding generation, feature extraction, and artifact creation.
- **Online Phase (`rank.py`)**: A Two-Stage retrieval pipeline. It uses fast matrix multiplication (Bi-Encoder) to filter the Top 500 candidates, then applies a Cross-Encoder for deep semantic reranking. Finally, it uses a 3-pillar blend scoring system to output the top 100 candidates to CSV.

## 2. Tech Stack & Technologies Used

### Programming Language
- **Python 3**: The entire project is written in Python.

### Machine Learning & NLP
- **PyTorch (`torch>=2.9.0`)**: Used for fast matrix multiplication and tensor operations (specifically for semantic search if a GPU/CUDA is available) and running the Cross-Encoder.
- **Sentence-Transformers**: Used to generate dense vector embeddings from text (career history, skills, job descriptions).
- **Transformers (Hugging Face)**: Underlying library for the models.
- **Models**: 
  - Embedding Model (Bi-Encoder): `BAAI/bge-small-en-v1.5` (33M parameters, 384-dim)
  - Cross-Encoder: `BAAI/bge-reranker-base`

### Vector Search & Information Retrieval
- **FAISS (`faiss-cpu`)**: A library for efficient similarity search and clustering of dense vectors.
- **BM25 (`rank_bm25`)**: A ranking function used for keyword-based/lexical search.

### Data Processing & Utilities
- **NumPy**: Used extensively for array manipulations, vector operations, and saving/loading embeddings (`.npy` files).
- **Pickle / JSON**: Used for serializing candidate data, extracted features, and index structures.
- **GZIP**: Used to handle compressed candidate data logs (`.jsonl.gz`).

## 3. Core Logic & Architecture

The ranking logic operates on a **3-Pillar Blend Scoring** methodology:
1. **Rule-Based Score (30%)**: Evaluates production experience, code recency, and ranking/search domain experience. Includes penalties for pure consulting backgrounds, and employs a softened hard-gate (requiring *both* zero production and zero recency to disqualify).
2. **Semantic Score (30%)**: Combines two stages:
   - *Stage 1 (Bi-Encoder)*: Fast cosine similarity using `bge-small-en-v1.5` embeddings, dynamically blending career and skills scores based on narrative authenticity.
   - *Stage 2 (Cross-Encoder)*: For the Top 500 candidates, the top-3 most relevant jobs are dynamically selected via keyword-relevance scoring (weighted by log-duration) and passed through the locally-cached `bge-reranker-base` model against the JD. The normalized cross-encoder score is dynamically blended with Stage 1 based on narrative authenticity (CE receives higher weight for synthetic profiles).
3. **Credibility Score (40%)**: A complex anti-stuffer mechanism that applies penalties for title-chasing, skill inflation, or failing honeypot checks (e.g., impossible career timelines). Also includes a strict Domain Penalty for purely CV/Speech candidates.

Additionally, behavioral multipliers (based on GitHub activity, profile completeness, assessment scores) and logistics constraints (relocation, work mode) are applied to the final score.

## 4. File Breakdown

| File Name | Purpose & Details |
|-----------|-------------------|
| `config.py` | Central configuration file defining paths (supports Colab and local), hyperparameters (e.g., `TOP_K_STAGE_1 = 500`), and model names. |
| `precompute.py` | The offline engine. Extracts candidate features, identifies synthetic/filler descriptions, computes embeddings using `SentenceTransformer`, pre-downloads the Cross-Encoder model to `artifacts/ce_model`, and saves `.npy` and `.json` artifacts. |
| `rank.py` | The online engine. Executes the Two-Stage Retrieval pipeline: filters Top 500 via Bi-Encoder, reranks via Cross-Encoder with keyword-scored smart job selection (truncating to reserve tokens for JD), applies the 3-pillar scoring logic, and outputs to `submission.csv`. |
| `honeypot_checks.py` | Contains 18+ strict rules to catch inflated or logically impossible resumes (e.g., overlapping full-time careers, post-grad experience impossible math). |
| `behavioral_signals.py` | Evaluates platform-level metadata such as profile views, recruiter response rates, GitHub activity, and assessment scores. |
| `reasoning_generator.py` | Constructs a JD-specific reasoning string (highlighting production retrieval experience and behavioral readiness) for the final CSV output. |
| `validate_submission.py` | A validation script to ensure the generated `submission.csv` strictly adheres to competition formatting rules. |
| `evaluate.py` | Offline evaluation script computing Precision@K, NDCG@K, Honeypot Leak Rate, and Score Separation using pseudo-labels to measure accuracy without ground truth. |
| `requirements.txt` | Lists the Python dependencies. |
| `commands.txt` | Quick reference guide for commands to set up the environment and run the pipeline. |

## 5. Execution Flow

1. **Setup**: Create a Python virtual environment and install dependencies (`pip install -r requirements.txt`).
2. **Phase 1 (Offline)**: Run `python precompute.py`. This creates an `artifacts/` folder populated with 384-dim `.npy` embeddings, `.json`, and `.pkl` files.
3. **Phase 2 (Online)**: Run `python rank.py` (< 5 minutes). This loads the artifacts, executes the Two-Stage ranking, and generates the final `submission.csv`.
4. **Validation/Evaluation**: Run `python validate_submission.py submission.csv` to ensure format correctness, and `python evaluate.py` to measure ranking quality.
