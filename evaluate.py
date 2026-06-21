#!/usr/bin/env python3
"""
evaluate.py  —  Offline Accuracy Evaluation for the Candidate Ranking System
=============================================================================

Since there are no human labels, we use HEURISTIC PSEUDO-LABELS derived from
features already computed in precompute.py. A candidate is deemed "relevant"
if they satisfy strong domain-specific criteria the JD explicitly requires.

Metrics computed:
  1. Precision@K           — % of top-K that are heuristically relevant
  2. NDCG@K                — graded relevance version (0/1/2 grades)
  3. Honeypot Leak Rate     — any disqualified candidates in top-100?
  4. Research Title Rate    — pure researchers (no production) in top-10?
  5. Score Distribution     — mean, std, percentiles (checks score spread)
  6. Score Separation       — gap between top-10 and bottom-10 of top-100
  7. Rank-Feature Correlation — do higher-ranked candidates have better features?
  8. Domain Signal Rates    — production exp / recency / ranking exp in top-K

Usage:
    python evaluate.py
    python evaluate.py --submission submission.csv --topk 10 20 50 100
"""

import os
import json
import csv
import math
import argparse
import pickle
from collections import defaultdict

import config

# ─────────────────────────────────────────────────────────────────────────────
# GRADED RELEVANCE  (0 = irrelevant, 1 = marginal, 2 = strong)
# ─────────────────────────────────────────────────────────────────────────────
def compute_relevance_grade(cid, features):
    """
    Assigns a relevance grade to a candidate based on pre-computed features.

    Grade 2 (Strong) — Meets all JD core requirements:
        - production_score >= 1   (has shipped to prod)
        - code_recency_score >= 1  (actively coding in 2025/2026)
        - ranking_score >= 2       (ranking/search/retrieval experience)
        - NOT a hard honeypot

    Grade 1 (Marginal) — Partially meets requirements:
        - Has some production experience OR ranking experience
        - Is NOT a hard honeypot

    Grade 0 (Irrelevant) — Fails minimum bar:
        - Hard honeypot candidate
        - Zero production experience AND zero recency
    """
    feat = features.get(cid, {})
    hp   = feat.get('honeypot_flags', {})

    if hp.get('hard', False):
        return 0   # Disqualified

    prod    = feat.get('production_score', 0)
    recency = feat.get('code_recency_score', 0)
    rank_xp = feat.get('ranking_score', 0)

    if prod >= 1 and recency >= 1 and rank_xp >= 2:
        return 2   # Strong match
    elif prod >= 1 or rank_xp >= 1:
        return 1   # Marginal
    else:
        return 0   # Irrelevant

# ─────────────────────────────────────────────────────────────────────────────
# NDCG COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
def dcg_at_k(grades, k):
    """Compute DCG@K for a ranked list of grades."""
    grades = grades[:k]
    return sum(
        (2**g - 1) / math.log2(i + 2)   # i+2 because log2(rank) starts at 1
        for i, g in enumerate(grades)
    )

def ndcg_at_k(grades, ideal_grades, k):
    """Compute NDCG@K: normalized DCG vs the ideal ordering."""
    dcg      = dcg_at_k(grades, k)
    ideal_dcg = dcg_at_k(sorted(ideal_grades, reverse=True), k)
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_submission(path):
    """Returns list of dicts: [{candidate_id, rank, score, reasoning}, ...]"""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['rank']  = int(row['rank'])
            row['score'] = float(row['score'])
            rows.append(row)
    rows.sort(key=lambda x: x['rank'])
    return rows

def load_candidates_lookup(cdata):
    """Build cid -> candidate dict from pickled list."""
    return {c['candidate_id']: c for c in cdata}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(submission_path, topk_list):

    # ── Load artifacts ──────────────────────────────────────────────────────
    print("Loading artifacts...")

    if not os.path.exists(config.FEATURES_PATH):
        print(f"❌ ERROR: features.json not found at {config.FEATURES_PATH}")
        print("   Run precompute.py first.")
        return

    with open(config.FEATURES_PATH, "r") as f:
        features = json.load(f)

    if not os.path.exists(config.CANDIDATE_DATA_PATH):
        print(f"❌ ERROR: candidate_data.pkl not found at {config.CANDIDATE_DATA_PATH}")
        print("   Run precompute.py first.")
        return

    with open(config.CANDIDATE_DATA_PATH, "rb") as f:
        cdata = pickle.load(f)

    cand_lookup = load_candidates_lookup(cdata)
    total_candidates = len(cdata)

    if not os.path.exists(submission_path):
        print(f"❌ ERROR: Submission not found at {submission_path}")
        print("   Run rank.py first.")
        return

    submission = load_submission(submission_path)
    print(f"✅ Loaded {len(submission)} ranked candidates from submission.")
    print(f"   Total candidate pool: {total_candidates:,}\n")

    # ── Compute relevance grades for all 100 submitted candidates ────────────
    submitted_ids   = [row['candidate_id'] for row in submission]
    submitted_scores = [row['score'] for row in submission]
    grades = [compute_relevance_grade(cid, features) for cid in submitted_ids]

    # Ideal grade list: best possible ordering from full pool
    # (sample 10k to avoid being too slow; statistically sufficient)
    sample_size = min(10000, total_candidates)
    import random
    random.seed(42)
    sample_ids = random.sample(list(cand_lookup.keys()), sample_size)
    ideal_grades = sorted(
        [compute_relevance_grade(cid, features) for cid in sample_ids],
        reverse=True
    )

    # ── Print separator ──────────────────────────────────────────────────────
    sep = "=" * 60

    print(sep)
    print("  RANKING EVALUATION REPORT")
    print(sep)

    # 1. PRECISION@K ──────────────────────────────────────────────────────────
    print("\n📊 1. PRECISION@K  (% of top-K that are heuristically relevant)")
    print(f"   {'K':<6} {'Strong (Gr.2)':<18} {'Marginal+ (Gr.1+)':<22} {'Irrelevant'}")
    print("   " + "-"*56)
    for k in topk_list:
        top_grades = grades[:k]
        strong   = sum(1 for g in top_grades if g == 2) / k
        marginal = sum(1 for g in top_grades if g >= 1) / k
        irrel    = sum(1 for g in top_grades if g == 0) / k
        print(f"   {k:<6} {strong*100:>6.1f}%           {marginal*100:>6.1f}%              {irrel*100:>6.1f}%")

    # 2. NDCG@K ───────────────────────────────────────────────────────────────
    print("\n📊 2. NDCG@K  (graded ranking quality; 1.0 = perfect)")
    print(f"   {'K':<6} {'NDCG'}")
    print("   " + "-"*18)
    for k in topk_list:
        ndcg = ndcg_at_k(grades, ideal_grades, k)
        bar = "█" * int(ndcg * 20)
        print(f"   {k:<6} {ndcg:.4f}  {bar}")

    # 3. HONEYPOT LEAK RATE ────────────────────────────────────────────────────
    print("\n🍯 3. HONEYPOT LEAK RATE  (hard-disqualified candidates in top-100)")
    leaks = []
    for row in submission:
        cid  = row['candidate_id']
        feat = features.get(cid, {})
        hp   = feat.get('honeypot_flags', {})
        if hp.get('hard', False):
            leaks.append((row['rank'], cid, hp.get('flags', [])))

    if leaks:
        print(f"   ⚠️  {len(leaks)} HONEYPOT CANDIDATES leaked into top-100!")
        for rank, cid, flags in leaks:
            print(f"      Rank {rank:>3}: {cid} — Flags: {flags}")
    else:
        print("   ✅ No hard-honeypot candidates in top-100. Clean.")

    # 4. RESEARCH TITLE RATE ───────────────────────────────────────────────────
    print("\n🔬 4. RESEARCH TITLE RATE  (pure researchers in top-10)")
    research_only = []
    for row in submission[:10]:
        cid  = row['candidate_id']
        c    = cand_lookup.get(cid, {})
        feat = features.get(cid, {})
        prod_score  = feat.get('production_score', 0)
        research_p  = feat.get('research_penalty', 1.0)
        title = c.get('profile', {}).get('current_title', 'Unknown')
        if prod_score == 0 or research_p < 0.9:
            research_only.append((row['rank'], cid, title, prod_score, research_p))

    if research_only:
        print(f"   ⚠️  {len(research_only)} potential pure-researchers in top-10:")
        for rank, cid, title, prod, rpen in research_only:
            print(f"      Rank {rank:>2}: {title:<40} prod={prod} research_penalty={rpen:.2f}")
    else:
        print("   ✅ Top-10 are free of pure researchers without production experience.")

    # 5. SCORE DISTRIBUTION ────────────────────────────────────────────────────
    print("\n📈 5. SCORE DISTRIBUTION  (top-100 scores)")
    scores = submitted_scores
    mean_s = sum(scores) / len(scores)
    variance = sum((s - mean_s)**2 for s in scores) / len(scores)
    std_s = variance**0.5
    sorted_s = sorted(scores, reverse=True)
    p25  = sorted_s[24]
    p50  = sorted_s[49]
    p75  = sorted_s[74]

    print(f"   Max   : {sorted_s[0]:.4f}   (Rank 1)")
    print(f"   P25   : {p25:.4f}   (Rank 25)")
    print(f"   Median: {p50:.4f}   (Rank 50)")
    print(f"   P75   : {p75:.4f}   (Rank 75)")
    print(f"   Min   : {sorted_s[-1]:.4f}   (Rank 100)")
    print(f"   Mean  : {mean_s:.4f}")
    print(f"   Std   : {std_s:.4f}  (higher = better separation)")

    # 6. SCORE SEPARATION ─────────────────────────────────────────────────────
    print("\n📏 6. SCORE SEPARATION  (top-10 vs bottom-10 of top-100)")
    top10_mean    = sum(scores[:10]) / 10
    bottom10_mean = sum(scores[-10:]) / 10
    separation    = top10_mean - bottom10_mean
    print(f"   Top-10  avg score : {top10_mean:.4f}")
    print(f"   Bottom-10 avg score: {bottom10_mean:.4f}")
    print(f"   Separation gap    : {separation:.4f}  (higher = better discrimination)")
    if separation < 0.02:
        print("   ⚠️  WARNING: Very low separation — scores are compressed.")
    elif separation > 0.1:
        print("   ✅ Good separation between top and bottom of the ranked list.")
    else:
        print("   ✅ Moderate separation. Acceptable.")

    # 7. RANK-FEATURE CORRELATION ─────────────────────────────────────────────
    print("\n🔗 7. RANK-FEATURE CORRELATION  (do higher ranks have better features?)")
    feature_keys = ['production_score', 'code_recency_score', 'ranking_score',
                    'behavioral_multiplier', 'narrative_authenticity']

    def pearson(x_list, y_list):
        n = len(x_list)
        if n == 0:
            return 0.0
        mx = sum(x_list)/n
        my = sum(y_list)/n
        num   = sum((x - mx)*(y - my) for x, y in zip(x_list, y_list))
        denom = (sum((x-mx)**2 for x in x_list) * sum((y-my)**2 for y in y_list))**0.5
        return num / denom if denom > 0 else 0.0

    ranks_list = list(range(1, 101))
    print(f"   {'Feature':<35} {'Corr with Rank':<20}  {'Direction'}")
    print("   " + "-"*65)
    for fk in feature_keys:
        feat_vals = [features.get(cid, {}).get(fk, 0) for cid in submitted_ids]
        corr = pearson(ranks_list, feat_vals)
        # Rank 1 = best, so negative correlation is GOOD (rank↑ => feature↑)
        direction = "✅ Good (rank↓ = feature↑)" if corr < -0.1 else \
                    "❌ Weak / Reversed"          if corr > 0.1 else \
                    "➖ Neutral"
        print(f"   {fk:<35} {corr:>+.4f}              {direction}")

    # 8. DOMAIN SIGNAL RATES ──────────────────────────────────────────────────
    print("\n🎯 8. DOMAIN SIGNAL RATES  (% of top-K with strong domain signals)")
    print(f"   {'K':<6} {'Prod. Exp.':<15} {'Recent Coding':<18} {'Rank Xp (>=2)'}")
    print("   " + "-"*55)
    for k in topk_list:
        top_ids = submitted_ids[:k]
        prod_rate   = sum(1 for cid in top_ids if features.get(cid,{}).get('production_score',0) >= 1) / k
        recency_rate = sum(1 for cid in top_ids if features.get(cid,{}).get('code_recency_score',0) >= 1) / k
        rank_rate   = sum(1 for cid in top_ids if features.get(cid,{}).get('ranking_score',0) >= 2) / k
        print(f"   {k:<6} {prod_rate*100:>6.1f}%        {recency_rate*100:>6.1f}%          {rank_rate*100:>6.1f}%")

    print(f"\n{sep}")
    print("  EVALUATION COMPLETE")
    print(sep)
    print()

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ranking quality offline.")
    parser.add_argument(
        "--submission", default=config.SUBMISSION_PATH,
        help="Path to the submission CSV (default: submission.csv)"
    )
    parser.add_argument(
        "--topk", nargs="+", type=int, default=[1, 5, 10, 25, 50, 100],
        help="Values of K for Precision@K and NDCG@K"
    )
    args = parser.parse_args()

    evaluate(args.submission, args.topk)
