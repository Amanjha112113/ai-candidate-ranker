# India Runs by Redrob AI Hackathon Submission

## Architecture

Our approach abandons pure semantic search in favor of a **3-Pillar Blend**: Rule-Based JD Compliance, Semantic Relevance, and Anti-Keyword-Stuffer Credibility.

### Phase 1: Pre-Computation (`precompute.py`)
Runs entirely offline to extract robust features and perform dataset-wide checks:
- **Strict JD Rule Extraction**: Computes hard disqualifiers (e.g., must have production experience, must have recent coding).
- **Skill Credibility**: Cross-references every claimed skill against actual job descriptions. Penalizes "keyword stuffers".
- **Honeypot Validation**: Runs 16 calibrated checks + H17 (Skill-Job Mismatch) + H18 (Rare Skill Combos).
- **Behavioral Scoring**: Computes exactly 5 composite multipliers (Availability, Market Validation, Trust, Reliability, Technical) from the 23 behavioral signals.
- **Artifact Generation**: Saves dense embeddings (`BAAI/bge-small-en-v1.5`) and feature JSONs.

### Phase 2: Online Ranking (`rank.py`)
Runs strictly under the 5-minute CPU-only constraint (typically < 5 seconds):
- **Stage 1 (Semantic FAISS)**: High-speed dot-product retrieval of candidate embeddings vs the JD intent vector.
- **Final Fit Blending (3-Pillar System)**: 
  - `30% Rule-Based Score` (Enforces hard JD disqualifiers)
  - `30% Semantic Score` (Cosine Similarity)
  - `40% Credibility Score` (Anti-stuffer penalty, tech credibility, honeypot multipliers)
- **Output**: Generates `submission.csv` strictly sorted by blended score.

## Setup Instructions
1. Install dependencies: `pip install -r requirements.txt`
2. Run Offline Feature Extraction (no time limit): `python precompute.py`
3. Run Online Ranking (<5 mins): `python rank.py`
