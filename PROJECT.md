# Project Discovery: Intelligent Candidate Discovery & Ranking Challenge

## 1. Problem Statement
The goal is to build an offline ranking system to identify the top 100 candidates for a "Senior AI Engineer — Founding Team" role out of a noisy pool of 100,000 candidates. The system must factor in not just semantic skill matching but also 23 behavioral signals (availability, market validation, reliability, trust, technical credibility) and rigorously detect "honeypot" candidates (fake/impossible profiles designed as traps). 

## 2. Input Format
- **`candidates.jsonl`**: A 100K-line JSONL file containing candidate profiles (ID, profile summary, career history, education, skills, certifications, and `redrob_signals`).
- **`job_description.md`**: The target JD for a Senior AI Engineer.
- **`redrob_signals_doc.md`**: Definition of 23 behavioral signals and their interpretation.
- **`Honeypot Detection System.pdf`**: Definition of 16 calibrated checks (hard/soft) to filter out fake profiles.

## 3. Output Format
- A single `submission.csv` file.
- Exactly 101 rows (1 header + 100 candidates).
- Columns: `candidate_id, rank, score, reasoning`.
- `rank` must be 1 to 100 without duplicates.
- `score` must be monotonically non-increasing.
- `reasoning` (optional but recommended): 1-2 sentence specific justification for the candidate's rank.

## 4. Evaluation Criteria
- Metrics computed against hidden ground truth:
  - 50% NDCG@10
  - 30% NDCG@50
  - 15% MAP (Mean Average Precision)
  - 5% P@10
- Disqualification rule: >10% honeypots in the top 100.
- Manual Review (Stage 4): Evaluating the `reasoning` column for specificity, JD connection, hallucination checks, and rank consistency.

## 5. Constraints
- **Runtime**: <= 5 minutes wall-clock for the ranking step.
- **Memory**: <= 16 GB RAM.
- **Compute**: CPU only (no GPU during ranking).
- **Network**: Completely offline (no external API calls during ranking).
- **Disk**: <= 5 GB intermediate state.
- **Pre-computation**: Allowed offline (e.g., generating embeddings, indexing).

## 6. Final Architecture Implementation
We have successfully implemented a **3-Pillar Candidate Ranking Engine**. All hardcoded keyword lists have been eliminated or integrated into a robust rule-based scoring module. The final blend heavily favors Structural Verification, Narrative Authenticity (detecting synthetic filler), and Semantic Precision (NDCG@10 optimized). Speed bottlenecks were resolved by pre-computing dual embeddings (Career/Skills) and replacing Cross-Encoders with high-speed NumPy dot products online, scoring 100K candidates in under 5 seconds.
