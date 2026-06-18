import os
import json
import gzip
import time
import pickle
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import gc

import config
from honeypot_checks import run_honeypot_checks
from behavioral_signals import compute_behavioral_signals
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# JD INTENT - Single Source of Truth
# ─────────────────────────────────────────────────────────────────────────────
JD_INTENT = (
    "Senior AI Engineer with production experience in embeddings-based retrieval systems. "
    "Deep expertise in vector databases (Pinecone, Weaviate, FAISS, Milvus, Qdrant). "
    "Proficiency in Python, PyTorch, and Transformers. "
    "Has shipped ranking, recommendation, or search systems to real users at scale. "
    "Familiar with evaluation frameworks: NDCG, MAP, MRR, A/B testing. "
    "Experience with LLM pipelines, RAG, and retrieval-augmented generation. "
    "Founding team mindset: ownership, urgency, iterates fast with data."
)

JD_HASH_PATH = os.path.join(config.ARTIFACTS_DIR, "jd_hash.txt")
JD_EMB_PATH  = os.path.join(config.ARTIFACTS_DIR, "jd_embedding.npy")

def get_jd_hash(jd_text):
    return hashlib.md5(jd_text.encode()).hexdigest()

def load_or_encode_jd(model, jd_text):
    current_hash = get_jd_hash(jd_text)
    if os.path.exists(JD_HASH_PATH) and os.path.exists(JD_EMB_PATH):
        with open(JD_HASH_PATH, "r") as f:
            stored_hash = f.read().strip()
        if stored_hash == current_hash:
            print("✓ JD hash matches — loading cached JD embedding (0.5s saved).")
            return np.load(JD_EMB_PATH)
    print("JD changed or first run — encoding JD query vector...")
    jd_emb = model.encode(
        ["Represent this sentence for searching relevant passages: " + jd_text],
        normalize_embeddings=True
    )
    np.save(JD_EMB_PATH, jd_emb)
    with open(JD_HASH_PATH, "w") as f:
        f.write(current_hash)
    return jd_emb

# ─────────────────────────────────────────────────────────────────────────────
# 1. PRODUCTION EXPERIENCE
# ─────────────────────────────────────────────────────────────────────────────
def calculate_production_score(candidate):
    engineering_titles = ['engineer', 'developer', 'scientist', 'architect', 'programmer']
    production_signals = ['production', 'shipped', 'deployed', 'launched', 'released', 
                         'scaling', 'scale', 'users', 'customers', 'real-world']
    research_titles = ['researcher', 'research scientist', 'research engineer']
    
    score = 0
    for job in candidate.get('career_history', []):
        title = job.get('title', '').lower()
        desc = job.get('description', '').lower()
        
        is_engineering = any(t in title for t in engineering_titles)
        if not is_engineering:
            continue
            
        if any(signal in desc for signal in production_signals):
            score += 1
            
    if score == 0:
        for job in candidate.get('career_history', []):
            if any(t in job.get('title', '').lower() for t in research_titles):
                return 0 # Research only
    return score

# ─────────────────────────────────────────────────────────────────────────────
# 2. CODE RECENCY
# ─────────────────────────────────────────────────────────────────────────────
def calculate_recency_score(candidate):
    engineering_titles = ['engineer', 'developer', 'scientist', 'architect', 'programmer']
    for job in candidate.get('career_history', []):
        title = job.get('title', '').lower()
        is_current = job.get('is_current', False)
        end_date = job.get('end_date')
        
        if is_current or (end_date and end_date.startswith("2026") or end_date.startswith("2025")):
            if any(t in title for t in engineering_titles):
                return 1
    return 0

# ─────────────────────────────────────────────────────────────────────────────
# 3. CONSULTING PENALTY
# ─────────────────────────────────────────────────────────────────────────────
def calculate_consulting_penalty(candidate):
    consulting_firms = ['tcs', 'wipro', 'infosys', 'accenture', 'cognizant', 'capgemini']
    total_jobs = len(candidate.get('career_history', []))
    if total_jobs == 0:
        return 1.0
        
    consulting_count = 0
    for job in candidate.get('career_history', []):
        comp = job.get('company', '').lower()
        if any(firm in comp for firm in consulting_firms):
            consulting_count += 1
            
    if consulting_count == total_jobs and total_jobs > 0:
        return 0.5 # Severe penalty if ONLY consulting
    return 1.0

# ─────────────────────────────────────────────────────────────────────────────
# 4. RANKING EXPERIENCE
# ─────────────────────────────────────────────────────────────────────────────
def calculate_ranking_score(candidate):
    keywords = {
        'ranking': ['ranking', 'rank', 'relevance', 're-rank', 'rerank', 'learn to rank', 'ltr'],
        'search': ['search', 'retrieval', 'retrieve', 'information retrieval', 'elasticsearch', 'opensearch'],
        'recommendation': ['recommendation', 'recommend', 'personalization', 'personalized', 'discovery'],
        'retrieval': ['retrieval', 'vector search', 'dense retrieval', 'hybrid retrieval', 'embedding']
    }
    
    score = 0
    engineering_titles = ['engineer', 'developer', 'scientist', 'architect']
    
    for job in candidate.get('career_history', []):
        title = job.get('title', '').lower()
        desc = job.get('description', '').lower()
        
        if not any(t in title for t in engineering_titles):
            continue
            
        for category, terms in keywords.items():
            if any(term in title for term in terms):
                score += 2.0
                break
                
        desc_score = 0
        for terms in keywords.values():
            if any(term in desc for term in terms):
                desc_score += 1.0
                
        duration_factor = min(job.get('duration_months', 0) / 24.0, 1.5)
        score += desc_score * duration_factor
        
    return min(score, 10.0)

# ─────────────────────────────────────────────────────────────────────────────
# 5a. FILLER TEMPLATE DETECTION & NARRATIVE AUTHENTICITY
# ─────────────────────────────────────────────────────────────────────────────
def build_filler_templates(candidates):
    """
    Scan all candidates and identify description templates that are reused
    across many distinct candidate_ids. These are "filler" descriptions from
    the synthetic data generator's template bank — they carry no real signal
    about a candidate's actual experience.
    """
    # Count how many DISTINCT candidates each normalized description appears in
    desc_candidate_count = Counter()
    for c in candidates:
        seen_descs = set()
        for job in c.get('career_history', []):
            desc = job.get('description', '').strip()
            if desc:
                # Normalize: lowercase, first 100 chars to catch minor variations
                norm = desc.lower()[:100]
                if norm not in seen_descs:
                    desc_candidate_count[norm] += 1
                    seen_descs.add(norm)
    
    # Threshold: any description appearing in >20 distinct candidates is filler.
    # This is conservative — real unique descriptions appear in 1 candidate.
    # Also compute a dynamic threshold: top 1% most frequent descriptions.
    total_unique = len(desc_candidate_count)
    static_threshold = 20
    
    # Use the stricter of: static threshold OR top 1% frequency
    if total_unique > 0:
        sorted_counts = sorted(desc_candidate_count.values(), reverse=True)
        dynamic_threshold = sorted_counts[max(0, int(total_unique * 0.01))] if total_unique > 100 else static_threshold
        threshold = min(static_threshold, dynamic_threshold)
    else:
        threshold = static_threshold
    
    filler = {desc for desc, count in desc_candidate_count.items() if count >= threshold}
    print(f"  Filler detection: {len(filler)} template descriptions identified (threshold={threshold})")
    print(f"  Total unique description prefixes: {total_unique}")
    return filler

def narrative_authenticity(candidate, filler_templates):
    """
    Compute how much of a candidate's career history consists of genuine,
    unique descriptions vs. filler templates from the synthetic data bank.
    
    Returns a float in [0.2, 1.0] that should be used as a multiplier on
    how much we trust the career-history semantic similarity score.
    
    Special handling: the most-recent / is_current role gets extra weight,
    since a genuine current ML role is the strongest positive signal.
    """
    careers = candidate.get('career_history', [])
    if not careers:
        return 0.3  # No career history at all = low trust
    
    total_entries = len(careers)
    filler_count = 0
    current_is_filler = False
    
    for job in careers:
        desc = job.get('description', '').strip()
        if not desc:
            filler_count += 1  # Empty description = treat as filler
            if job.get('is_current', False):
                current_is_filler = True
            continue
        norm = desc.lower()[:100]
        if norm in filler_templates:
            filler_count += 1
            if job.get('is_current', False):
                current_is_filler = True
    
    filler_ratio = filler_count / total_entries
    
    # Base authenticity from filler ratio
    if filler_ratio < 0.3:
        base = 1.0   # Mostly unique descriptions — full trust
    elif filler_ratio < 0.6:
        base = 0.7   # Mixed — moderate trust
    else:
        base = 0.4   # Mostly filler — low trust
    
    # Extra penalty if the CURRENT role is filler (most important role)
    if current_is_filler:
        base *= 0.7
    
    return max(0.2, base)  # Floor at 0.2, never fully zero

# ─────────────────────────────────────────────────────────────────────────────
# 5b. CAREER CONSISTENCY
# ─────────────────────────────────────────────────────────────────────────────
def calculate_career_consistency(candidate):
    industries = [job.get('industry', '') for job in candidate.get('career_history', [])]
    ml_industries = {'software', 'ai/ml', 'it services', 'fintech', 'e-commerce', 'transportation', 'food delivery', 'technology', 'internet'}
    ml_related = sum(1 for ind in industries if any(kw in ind.lower() for kw in ml_industries))
    if len(industries) >= 3 and ml_related / len(industries) < 0.5:
        return 0.8
    return 1.0

# ─────────────────────────────────────────────────────────────────────────────
# 6. TECHNICAL CREDIBILITY BOOST
# ─────────────────────────────────────────────────────────────────────────────
def calculate_tech_credibility(candidate):
    signals = candidate.get('redrob_signals', {})
    
    github_score = signals.get('github_activity_score', -1)
    if github_score < 0:
        github_boost = 0.3
    elif github_score < 20:
        github_boost = 0.8
    else:
        github_boost = min(1.2, 1.0 + (github_score / 100.0))
        
    assessment_count = len(signals.get('skill_assessment_scores', {}))
    if assessment_count == 0:
        assessment_boost = 0.5
    elif assessment_count < 3:
        assessment_boost = 0.8
    else:
        assessment_boost = 1.0
        
    interview_rate = signals.get('interview_completion_rate', 0)
    if interview_rate > 0.8:
        interview_boost = 1.1
    elif interview_rate < 0.3:
        interview_boost = 0.7
    else:
        interview_boost = 1.0
        
    return github_boost * assessment_boost * interview_boost

# ─────────────────────────────────────────────────────────────────────────────
# 7. TITLE-CHASER & EXTERNAL VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def calculate_title_chaser_penalty(candidate):
    career = candidate.get('career_history', [])
    if len(career) < 3:
        return 1.0
    
    # Average tenure
    tenures = [job.get('duration_months', 0) for job in career if job.get('duration_months', 0) > 0]
    if not tenures:
        return 1.0
    avg_tenure = sum(tenures) / len(tenures)
    
    # Check for title inflation
    titles = [job.get('title', '').lower() for job in career]
    has_inflation = any('senior' in titles[i] and 'senior' not in titles[i-1] for i in range(1, len(titles)))
    
    if avg_tenure < 18 and has_inflation:
        return 0.8
    return 1.0

def calculate_external_validation_penalty(candidate):
    yoe = candidate.get('profile', {}).get('years_of_experience', 0)
    if yoe < 5:
        return 1.0
    
    sig = candidate.get('redrob_signals', {})
    github = sig.get('github_activity_score', -1)
    certs = candidate.get('certifications', [])
    
    # Proxy: if no GitHub activity and no certifications, penalize
    if github < 20 and len(certs) == 0:
        return 0.9
    return 1.0

# ─────────────────────────────────────────────────────────────────────────────
# 8. RESEARCH TITLE PENALTY (JD explicitly disqualifies pure research)
# ─────────────────────────────────────────────────────────────────────────────
def calculate_research_penalty(candidate):
    """
    The JD explicitly states: 'If you've spent your career in pure research 
    environments... without any production deployment — we will not move forward.'
    
    Penalizes candidates with 'research' in their titles who lack strong 
    production evidence. Does NOT penalize research titles that also have 
    clear production signals (they are valuable hybrids).
    """
    current_title = candidate.get('profile', {}).get('current_title', '').lower()
    career_titles = [job.get('title', '').lower() for job in candidate.get('career_history', [])]
    all_titles = career_titles + [current_title]
    
    # Count how many roles have "research" in the title
    research_count = sum(1 for t in all_titles if 'research' in t)
    total_roles = max(len(all_titles), 1)
    research_ratio = research_count / total_roles
    
    if research_ratio == 0:
        return 1.0  # No research titles at all — no penalty
    
    # Check for production evidence across career descriptions
    production_signals = ['production', 'shipped', 'deployed', 'launched', 'released',
                         'scaling', 'scale', 'users', 'customers', 'real-world',
                         'served', 'traffic', 'api', 'pipeline', 'microservice']
    
    production_evidence = 0
    for job in candidate.get('career_history', []):
        desc = job.get('description', '').lower()
        if any(sig in desc for sig in production_signals):
            production_evidence += 1
    
    if production_evidence >= 2:
        return 1.0   # Strong production evidence — hybrid researcher, no penalty
    elif production_evidence == 1:
        # Some production, but mostly research
        if research_ratio > 0.5:
            return 0.85
        return 0.95
    else:
        # No production evidence + research titles
        if research_ratio > 0.5:
            return 0.6   # Mostly research career — heavy penalty
        return 0.75      # Some research roles — moderate penalty

# ─────────────────────────────────────────────────────────────────────────────
# 9. PRODUCTION DEPTH BOOST (rewards multi-company production experience)
# ─────────────────────────────────────────────────────────────────────────────
def calculate_production_depth_boost(candidate):
    """
    Counts the number of distinct roles with production evidence.
    Candidates who shipped at 3+ companies (e.g., Swiggy, Uber, Zomato)
    should score significantly higher than someone with 1 production role.
    """
    production_signals = ['production', 'shipped', 'deployed', 'launched', 'released',
                         'scaling', 'scale', 'users', 'customers', 'real-world',
                         'served', 'traffic', 'api', 'pipeline', 'microservice',
                         'millions', 'latency', 'throughput', 'sla']
    
    production_roles = 0
    for job in candidate.get('career_history', []):
        desc = job.get('description', '').lower()
        if any(sig in desc for sig in production_signals):
            production_roles += 1
    
    if production_roles >= 3:
        return 1.15  # Strong multi-company production depth
    elif production_roles >= 2:
        return 1.08
    elif production_roles >= 1:
        return 1.0
    else:
        return 0.85  # No production evidence at all

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("Starting Pre-computation Phase (Strict JD Compliance)...")
    start_time = time.time()

    # ── Load candidates ──────────────────────────────────────────────────────
    candidates = []
    cand_path = config.CANDIDATES_PATH
    if os.path.exists(cand_path + ".gz"):
        with gzip.open(cand_path + ".gz", "rt") as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))
    else:
        with open(cand_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))
                    
    print(f"Loaded {len(candidates)} candidates.")

    # ── Global Skill Frequencies (for H18) ───────────────────────────────────
    global_skill_freq = {}
    for c in candidates:
        for sk in c.get('skills', []):
            name = sk.get('name', '').lower()
            if name:
                global_skill_freq[name] = global_skill_freq.get(name, 0) + 1

    # ── Build Filler Template Index ───────────────────────────────────────────
    print("Building filler template index across all candidates...")
    filler_templates = build_filler_templates(candidates)

    # ── Load model and encode JD ─────────────────────────────────────────────
    print("Loading Semantic Model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=device)
    encode_batch_size = 2048 if device == "cuda" else 256
    jd_emb = load_or_encode_jd(model, JD_INTENT)
    
    print("Encoding all unique skills for corroboration...")
    unique_skills = list(global_skill_freq.keys())
    skill_embs = model.encode(["Represent this document for retrieval: " + s for s in unique_skills], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    skill_emb_map = {s: skill_embs[i] for i, s in enumerate(unique_skills)}
    
    # ── Feature Extraction ───────────────────────────────────────────────────
    print("Extracting Candidate Features...")
    features = {}
    
    for idx, c in enumerate(candidates):
        cid = c["candidate_id"]
        
        # Honeypot checks
        hp = run_honeypot_checks(c, global_skill_freq)
        
        # Behavioral signals (compute ONCE)
        beh_signals = compute_behavioral_signals(c)
        
        # Rule-based Features
        feat = {
            "production_score": calculate_production_score(c),
            "code_recency_score": calculate_recency_score(c),
            "consulting_penalty": calculate_consulting_penalty(c),
            "ranking_score": calculate_ranking_score(c),
            "tech_credibility": calculate_tech_credibility(c),
            "career_consistency_penalty": calculate_career_consistency(c),
            "title_chaser_penalty": calculate_title_chaser_penalty(c),
            "external_validation_penalty": calculate_external_validation_penalty(c),
            "research_penalty": calculate_research_penalty(c),
            "production_depth_boost": calculate_production_depth_boost(c),
            "honeypot_flags": hp,
            "behavioral_multiplier": beh_signals.get("behavioral_multiplier", 1.0),
            "jd_fit_multiplier": beh_signals.get("jd_fit_multiplier", 1.0),
            "narrative_authenticity": narrative_authenticity(c, filler_templates)
        }
        features[cid] = feat
        
        if (idx + 1) % 10000 == 0:
            print(f"  Features extracted: {idx + 1}/{len(candidates)}")

    print(f"Feature extraction complete. ({time.time() - start_time:.1f}s elapsed)")

    # ── Build ALL text blobs first (memory: ~100MB for 200k strings) ──────────
    print("Building text blobs for embedding...")
    all_career_texts = []
    all_skills_texts = []
    
    for c in candidates:
        career_text = " ".join([job.get('description', '') for job in c.get('career_history', [])])[:500]
        adv_skills = [sk.get('name', '') for sk in c.get('skills', []) 
                     if sk.get('proficiency', '') in ['advanced', 'expert']][:5]
        headline = c.get('profile', {}).get('headline', '')
        summary = c.get('profile', {}).get('summary', '')[:500]
        skills_text = f"{headline} {summary} {' '.join(adv_skills)}"
        
        all_career_texts.append("Represent this document for retrieval: " + career_text)
        all_skills_texts.append("Represent this document for retrieval: " + skills_text)

    # ── Encode ALL career texts in ONE call (library handles internal batching) ─
    print(f"Encoding {len(all_career_texts)} career texts (this takes ~3-5 min on CPU)...")
    career_embeddings = model.encode(
        all_career_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=encode_batch_size
    ).astype(np.float16)
    
    print(f"Encoding {len(all_skills_texts)} skills texts...")
    skills_embeddings = model.encode(
        all_skills_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=encode_batch_size
    ).astype(np.float16)
    
    # ── Skill Corroboration (fast vectorized dot products) ────────────────────
    print("Computing skill corroboration scores...")
    for idx, c in enumerate(candidates):
        cid = c["candidate_id"]
        career_emb = career_embeddings[idx]
        
        career_text = all_career_texts[idx]
        if len(career_text) > 50:  # Has actual content beyond the prefix
            skill_sims = []
            for sk in c.get('skills', []):
                if sk.get('proficiency') in ['advanced', 'expert']:
                    sk_name = sk.get('name', '').lower()
                    if sk_name in skill_emb_map:
                        sim = float(np.dot(career_emb.astype(np.float32), skill_emb_map[sk_name]))
                        skill_sims.append(sim)
            if skill_sims:
                avg_corrob = np.mean(skill_sims)
                cred_penalty = min(1.0, avg_corrob / 0.4)
            else:
                cred_penalty = 1.0
        else:
            cred_penalty = 0.3
            
        features[cid]["skill_corroboration_penalty"] = cred_penalty

    # Free memory
    del all_career_texts
    del all_skills_texts
    gc.collect()

    # ── Save embeddings ──────────────────────────────────────────────────────
    print("Saving embeddings...")
    np.save(os.path.join(config.ARTIFACTS_DIR, "career_embeddings.npy"), career_embeddings)
    np.save(os.path.join(config.ARTIFACTS_DIR, "skills_embeddings.npy"), skills_embeddings)

    # ── Save all artifacts ───────────────────────────────────────────────────
    print("Saving artifacts...")
    with open(config.FEATURES_PATH, "w") as f:
        json.dump(features, f)
    with open(config.CANDIDATE_DATA_PATH, "wb") as f:
        pickle.dump(candidates, f)

    print(f"\n✅ Pre-computation complete in {time.time() - start_time:.2f}s.")

if __name__ == "__main__":
    main()
