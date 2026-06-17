# Architecture Report

## 1. Data Pipeline
The pipeline is strictly divided into an **Offline Pre-computation Phase** and an **Online Ranking Phase** to meet the 5-minute CPU-only constraint.

### Offline Pre-computation Phase (`precompute.py`)
1. **Candidate Ingestion**: Parse `candidates.jsonl`.
2. **Behavioral Signal Processing**: Compute the 23 behavioral signals into composite multipliers.
3. **Rule-Based Feature Extraction (JD Disqualifiers)**:
   - **Production Score**: Identifies "shipped", "deployed", "scale" paired with engineering titles. Disqualifies pure research.
   - **Code Recency Score**: Verifies coding experience in 2025/2026.
   - **Consulting Penalty**: Flags careers spent exclusively at consulting firms (TCS, Wipro, Infosys, etc.).
   - **Ranking Score**: Detects deep experience with ranking, retrieval, and recommendation.
4. **Credibility Validation (Anti-Stuffer)**:
   - **Skill Credibility**: Cross-references every claimed skill against actual job descriptions. Heavy penalty for unsupported skills.
   - **Technical Credibility**: Uses GitHub scores, assessment counts, and interview rates.
5. **SOTA Honeypot Validation (H01-H19)**:
   - Fixed H07 (Salary) to only penalize if paired with other red flags.
   - Added H17 (Skill-Job Mismatch) checking if <20% of skills match job titles.
   - Added H18 (Rare Skill Combo) using global dataset frequencies to flag impossible expert claims.
6. **Semantic Artifacts**:
   - Computes dense embeddings (`BAAI/bge-small-en-v1.5`) for core profile fields (title, summary, advanced skills).
   - FAISS Index built for ultra-fast online ranking.

### Online Ranking Phase (`rank.py`)
Runs completely offline in ~2 seconds (well under the 5-minute CPU-only constraint):
1. **Semantic Scoring**: FAISS/NumPy dot product of candidate embeddings vs. JD intent vector.
2. **3-Pillar Evaluation**:
   - **Pillar 1: Rule-Based Score (30%)**: Enforces hard JD disqualifiers (zero score if no production experience or no recent coding). Awards points for ranking experience and avoids consulting-only backgrounds.
   - **Pillar 2: Semantic Score (30%)**: Cosine similarity.
   - **Pillar 3: Credibility Score (40%)**: `Skill Credibility * Technical Credibility * Consulting Penalty * Honeypot Penalties`.
3. **Final Scoring**:
   - `Final = (0.3 * Rule + 0.3 * Semantic + 0.4 * Credibility) * Behavioral_Multiplier`
4. **Reasoning**: Generates transparent score breakdowns.

---

## 2. Flaw Resolution Log

| Flaw | Old Approach | Fixed Approach | Verdict |
|---|---|---|---|
| Skill Inflation | Blindly trusted durations | `calculate_skill_credibility` checks job descriptions for skill presence | ✅ Fixed |
| JD Interpretation | Heavy semantic embedding reliance | Strict rule-based Stage 1 with hard zeros for missing production/recency | ✅ Fixed |
| Ranking Experience | Narrow exact match | Semantic clusters + title weighting | ✅ Fixed |
| Reranking Speed | Cross-Encoder (~90s) | Replaced with FAISS dot product + 3-Pillar blend (~2s) | ✅ Fixed |
| H07 Salary Mismatch | Blind penalty on 18% of pool | Only penalizes if other soft flags are present | ✅ Fixed |
