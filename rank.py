import os
import time
import pickle
import json
import csv
import numpy as np
import torch
import math

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
    print("Executing Stage 1: Bi-Encoder Semantic Search...")
    t = time.time()
    if len(jd_emb.shape) == 1:
        jd_emb = jd_emb.reshape(1, -1)
        
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        device = 'cuda' if torch.cuda.is_available() else 'mps'
        print(f"Using {device.upper()} for fast matrix multiplication...")
        career_tensor = torch.tensor(career_embs, device=device, dtype=torch.float16 if device == 'cuda' else torch.float32)
        skills_tensor = torch.tensor(skills_embs, device=device, dtype=torch.float16 if device == 'cuda' else torch.float32)
        jd_tensor = torch.tensor(jd_emb.T, device=device, dtype=torch.float16 if device == 'cuda' else torch.float32)
        
        career_sims = torch.matmul(career_tensor, jd_tensor).cpu().numpy().flatten()
        skills_sims = torch.matmul(skills_tensor, jd_tensor).cpu().numpy().flatten()
        
        del career_tensor
        del skills_tensor
        del jd_tensor
    else:
        career_sims = np.dot(career_embs, jd_emb.T).flatten()
        skills_sims = np.dot(skills_embs, jd_emb.T).flatten()
        
    print(f"Semantic scoring complete in {time.time() - t:.2f}s")
    
    # ── Stage 1: Filtering Top 500 ──────────────────────────────────────────
    print(f"Executing Stage 1: Filtering Top {config.TOP_K_STAGE_1}...")
    t = time.time()
    
    stage1_scores = []
    
    for i, candidate in enumerate(cdata):
        cid = candidate["candidate_id"]
        feat = feat_map.get(cid, {})
        
        # ── Pillar 1: Rule-Based Score (0.0 to 1.0)
        prod_score = feat.get('production_score', 0)
        recency = feat.get('code_recency_score', 0)
        rank_exp = feat.get('ranking_score', 0)
        consult = feat.get('consulting_penalty', 1.0)
        
        rule_score = min(0.35, 0.15 * math.log1p(prod_score))
        if recency > 0: rule_score += 0.25
        rule_score += min(0.25, rank_exp * 0.05)
        if consult > 0.5: rule_score += 0.15
        # Hard gate: both production evidence AND recency must be zero to nullify.
        # Using OR was too aggressive — it zeroed out candidates with recent non-ML
        # engineering roles that still have ranking/search experience in descriptions.
        if prod_score == 0 and recency == 0: rule_score = 0.0
            
        # ── Pillar 2: Stage 1 Semantic Score (0.0 to 1.0)
        sem_score_career = (float(career_sims[i]) + 1) / 2.0
        sem_score_skills = (float(skills_sims[i]) + 1) / 2.0
        
        authenticity = feat.get('narrative_authenticity', 1.0)
        
        # Priority 3: Dynamic blending based on authenticity
        career_weight = 0.3 + (0.4 * authenticity)
        skills_weight = 1.0 - career_weight
        
        stage1_sem_score = (career_weight * sem_score_career) + (skills_weight * sem_score_skills)
        
        # ── Pillar 3: Credibility Score (Anti-Stuffer)
        cred_score = (
            feat.get('skill_corroboration_penalty', 1.0) * 
            feat.get('tech_credibility', 1.0) * 
            feat.get('consulting_penalty', 1.0) *
            feat.get('career_consistency_penalty', 1.0) *
            feat.get('domain_penalty', 1.0)
        )
        
        hp = feat.get('honeypot_flags', {})
        if hp.get('hard', False):
            cred_score = 0.0
            rule_score = 0.0
        else:
            cred_score *= hp.get('penalty', 1.0)
            
        cred_score *= feat.get('title_chaser_penalty', 1.0)
        cred_score *= feat.get('external_validation_penalty', 1.0)
        cred_score *= feat.get('research_penalty', 1.0)
        cred_score *= feat.get('production_depth_boost', 1.0)
            
        # Behavioral Multiplier
        beh_mult = feat.get('behavioral_multiplier', 1.0)
        jd_fit = feat.get('jd_fit_multiplier', 1.0)
            
        # Stage 1 Blend
        blend = (
            0.3 * rule_score +
            0.3 * stage1_sem_score +
            0.4 * cred_score
        )
        
        final_score = blend * beh_mult * jd_fit
        
        stage1_scores.append((final_score, cid, i, stage1_sem_score, rule_score, cred_score, beh_mult, jd_fit))
        
    print(f"Stage 1 scoring complete in {time.time() - t:.2f}s")
    
    stage1_scores.sort(key=lambda x: (-x[0], x[1]))
    top_500 = stage1_scores[:config.TOP_K_STAGE_1]
    
    # ── Stage 2: Cross-Encoder Reranking ────────────────────────────────────
    print(f"Executing Stage 2: Cross-Encoder Reranking for Top {config.TOP_K_STAGE_1}...")
    t = time.time()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    def get_relevant_jobs_text(candidate):
        """
        Select the 3 most JD-relevant job descriptions for the cross-encoder.
        Scores each role by keyword relevance (retrieval / ranking / production terms)
        weighted by duration, then returns the top 3 concatenated.
        
        Better than boolean sort: an ML engineer with 24mo of embedding work ranks
        above a data engineer with 1mo of vague 'search analytics' mentions.
        """
        JD_KEYWORDS = [
            'retrieval', 'ranking', 'search', 'recommendation', 'embedding', 'vector',
            'semantic', 'nlp', 'machine learning', 'deployed', 'production', 'shipped',
            'pipeline', 'faiss', 'pinecone', 'qdrant', 'weaviate', 'milvus',
            'elasticsearch', 'opensearch', 'llm', 'rag', 'transformer', 'pytorch'
        ]
        scored_jobs = []
        for job in candidate.get('career_history', []):
            desc = job.get('description', '').strip()
            if not desc:
                continue
            desc_lower = desc.lower()
            title_lower = job.get('title', '').lower()
            duration = job.get('duration_months', 1) or 1

            # Keyword hit count, weighted by log-duration for longer roles
            hits = sum(1 for kw in JD_KEYWORDS if kw in desc_lower or kw in title_lower)
            relevance = hits * math.log1p(duration)
            scored_jobs.append((relevance, desc[:400]))

        # Sort descending by relevance score, take top 3
        scored_jobs.sort(key=lambda x: -x[0])
        top_texts = [desc for _, desc in scored_jobs[:3]]
        return " ".join(top_texts)

    # Load cross-encoder model locally (cached during precompute)
    print("Loading Cross-Encoder model locally...")
    ce_path = os.path.join(config.ARTIFACTS_DIR, "ce_model")
    if not os.path.exists(ce_path):
        raise FileNotFoundError(
            f"Cross-encoder not cached at '{ce_path}'. "
            "Run precompute.py first — it downloads and saves the model offline."
        )
    tokenizer = AutoTokenizer.from_pretrained(ce_path)
    ce_model = AutoModelForSequenceClassification.from_pretrained(ce_path)
    ce_model.eval()
    ce_model.to(device)

    # Build (JD, smart_career_text) pairs for the Top 500
    print("Building text for cross-encoder input...")
    pairs = []
    for candidate_tuple in top_500:
        idx = candidate_tuple[2]
        c = cdata[idx]
        career_text = get_relevant_jobs_text(c)[:300]  # Reserve token budget for JD (~180 tokens)
        pairs.append([JD_INTENT, career_text])
        
    ce_logits = []
    batch_size = 32
    
    with torch.inference_mode():
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i+batch_size]
            inputs = tokenizer(batch_pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
            outputs = ce_model(**inputs)
            logits = outputs.logits.squeeze(-1).cpu().tolist()
            if isinstance(logits, float):
                ce_logits.append(logits)
            else:
                ce_logits.extend(logits)
                
    # Priority 2: Min-Max Scaling
    min_ce = min(ce_logits)
    max_ce = max(ce_logits)
    ce_range = max_ce - min_ce if max_ce > min_ce else 1.0
    ce_scores_normalized = [(score - min_ce) / ce_range for score in ce_logits]
    
    print(f"Stage 2 Cross-Encoder complete in {time.time() - t:.2f}s")
    
    # ── Final Blending ──────────────────────────────────────────────────────
    print("Executing Final Blend Scoring...")
    final_top_500 = []
    scores_map = {}
    
    for i, candidate_tuple in enumerate(top_500):
        _, cid, idx, stage1_sem_score, rule_score, cred_score, beh_mult, jd_fit = candidate_tuple
        
        ce_score = ce_scores_normalized[i]

        # Dynamic CE weight: when narrative_authenticity is LOW (synthetic career text),
        # the bi-encoder career embedding is unreliable — trust the CE more.
        # When authenticity is HIGH (genuine descriptions), both signals are reliable.
        feat = feat_map.get(cid, {})
        authenticity = feat.get('narrative_authenticity', 1.0)
        ce_weight = 0.5 + (0.3 * (1.0 - authenticity))  # CE weight: 0.50 (authentic) → 0.80 (filler)
        bi_weight = 1.0 - ce_weight
        sem_final = bi_weight * stage1_sem_score + ce_weight * ce_score
        
        blend = (
            0.3 * rule_score +
            0.3 * sem_final +
            0.4 * cred_score
        )
        
        final_score = blend * beh_mult * jd_fit
        rounded_score = round(final_score, 4)
        
        final_top_500.append((rounded_score, cid, idx))
        
        scores_map[idx] = {
            "final": rounded_score,
            "rule": rule_score,
            "semantic": sem_final,
            "credibility": cred_score,
            "behavioral": beh_mult
        }

    # Sort strictly: Score Descending, ID Ascending
    final_top_500.sort(key=lambda x: (-x[0], x[1]))
    top_100_idx = [idx for _, _, idx in final_top_500[:config.FINAL_TOP_K]]
    
    print(f"Writing {config.FINAL_TOP_K} results to CSV...")
    with open(config.SUBMISSION_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        
        for rank, idx in enumerate(top_100_idx, 1):
            c = cdata[idx]
            cid = c["candidate_id"]
            s = scores_map[idx]
            
            # Priority 4: Use actual reasoning generator
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
