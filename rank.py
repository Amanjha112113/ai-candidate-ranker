import os
import time
import pickle
import json
import csv
import numpy as np

import config
from reasoning_generator import generate_reasoning
from precompute import JD_INTENT, get_jd_hash

def load_artifacts():
    print("Loading artifacts...")
    t = time.time()
    
    with open(config.CANDIDATE_DATA_PATH, "rb") as f:
        cdata = pickle.load(f)
        
    career_embs = np.load(os.path.join(config.ARTIFACTS_DIR, "career_embeddings.npy"))
    skills_embs = np.load(os.path.join(config.ARTIFACTS_DIR, "skills_embeddings.npy"))
    
    with open(config.FEATURES_PATH, "r") as f:
        features = json.load(f)
        
    jd_emb = np.load(os.path.join(config.ARTIFACTS_DIR, "jd_embedding.npy"))
        
    print(f"Loaded artifacts in {time.time() - t:.2f}s")
    return cdata, career_embs, skills_embs, features, jd_emb

def main():
    total_start = time.time()
    
    cdata, career_embs, skills_embs, feat_map, jd_emb = load_artifacts()
    
    # ── Stage 1: Fast FAISS/NumPy Semantic Scoring ───────────────────────────
    print("Executing Semantic Search...")
    t = time.time()
    if len(jd_emb.shape) == 1:
        jd_emb = jd_emb.reshape(1, -1)
        
    career_sims = np.dot(career_embs, jd_emb.T).flatten()
    skills_sims = np.dot(skills_embs, jd_emb.T).flatten()
        
    print(f"Semantic scoring complete in {time.time() - t:.2f}s")
    
    # ── Stage 2: Final Blending ──────────────────────────────────────────────
    print("Executing 3-Pillar Blend Scoring...")
    t = time.time()
    
    final_scores = []
    scores_map = {}
    
    for i, candidate in enumerate(cdata):
        cid = candidate["candidate_id"]
        feat = feat_map.get(cid, {})
        
        # ── Pillar 1: Rule-Based Score (0.0 to 1.0)
        rule_score = 0.0
        if feat.get('production_score', 0) > 0:
            rule_score += 0.3
        if feat.get('code_recency_score', 0) > 0:
            rule_score += 0.3
        if feat.get('ranking_score', 0) > 0:
            rule_score += 0.2
        if feat.get('consulting_penalty', 1.0) > 0.5:
            rule_score += 0.2
            
        # Hard Disqualifiers - MUST HAVE production & recent coding
        if feat.get('production_score', 0) == 0 or feat.get('code_recency_score', 0) == 0:
            rule_score = 0.0
            
        # ── Pillar 2: Semantic Score (0.0 to 1.0)
        # Convert -1..1 to 0..1
        sem_score_career = (float(career_sims[i]) + 1) / 2.0
        sem_score_skills = (float(skills_sims[i]) + 1) / 2.0
        
        # Narrative authenticity gates how much we trust the career embedding.
        # If career descriptions are filler templates from the synthetic data bank,
        # they carry no real signal — discount career_sim, lean on skills/title.
        authenticity = feat.get('narrative_authenticity', 1.0)
        
        # Dynamic blending: high authenticity = trust career (70/30),
        # low authenticity = trust skills more (30/70)
        career_weight = 0.3 + 0.4 * authenticity  # ranges from 0.38 to 0.70
        skills_weight = 1.0 - career_weight         # ranges from 0.30 to 0.62
        sem_score = career_weight * sem_score_career + skills_weight * sem_score_skills
        
        # ── Pillar 3: Credibility Score (Anti-Stuffer)
        cred_score = (
            feat.get('skill_corroboration_penalty', 1.0) * 
            feat.get('tech_credibility', 1.0) * 
            feat.get('consulting_penalty', 1.0) *
            feat.get('career_consistency_penalty', 1.0)
        )
        
        # Honeypot Penalties
        hp = feat.get('honeypot_flags', {})
        if hp.get('hard', False):
            cred_score = 0.0
            rule_score = 0.0
        else:
            cred_score *= hp.get('penalty', 1.0)
            
        cred_score *= feat.get('title_chaser_penalty', 1.0)
        cred_score *= feat.get('external_validation_penalty', 1.0)
            
        # Behavioral Multiplier (Logistics)
        beh_mult = feat.get('behavioral_multiplier', 1.0)
        jd_fit = feat.get('jd_fit_multiplier', 1.0)
            
        # ── Final Blend (30% Rule, 30% Semantic, 40% Credibility)
        blend = (
            0.3 * rule_score +
            0.3 * sem_score +
            0.4 * cred_score
        )
        
        final_score = blend * beh_mult * jd_fit
        rounded_score = round(final_score, 4)
        
        scores_map[i] = {
            "final": rounded_score,
            "rule": rule_score,
            "semantic": sem_score,
            "credibility": cred_score,
            "behavioral": beh_mult
        }
        
        final_scores.append((rounded_score, cid, i))
        
    print(f"Scoring complete in {time.time() - t:.2f}s")
    
    # ── Sort and Output ──────────────────────────────────────────────────────
    # Sort strictly: Score Descending, ID Ascending
    final_scores.sort(key=lambda x: (-x[0], x[1]))
    
    top_100_idx = [idx for _, _, idx in final_scores[:config.FINAL_TOP_K]]
    
    print(f"Writing {config.FINAL_TOP_K} results to CSV...")
    with open(config.SUBMISSION_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        
        for rank, idx in enumerate(top_100_idx, 1):
            c = cdata[idx]
            cid = c["candidate_id"]
            s = scores_map[idx]
            
            # Use actual reasoning generator
            reasoning = generate_reasoning(c, rank, s)
            
            writer.writerow({
                "candidate_id": cid,
                "rank": rank,
                "score": f"{s['final']:.4f}",
                "reasoning": reasoning
            })
            
    print(f"✅ Success! Submission saved to {config.SUBMISSION_PATH}")
    print(f"Total Online Execution Time: {time.time() - total_start:.2f}s")

if __name__ == "__main__":
    main()
