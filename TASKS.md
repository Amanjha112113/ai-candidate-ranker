# Implementation Tasks

## High Priority
- [x] Setup the offline pipeline (`precompute.py`) using exactly 16 honeypot checks + SOTA Semantic AI checks.
- [x] Implement robust behavioral signal processing based on `behavioral_signals.md` formulas.
- [x] Implement semantic evaluation replacing hardcoded strings (BGE model scoring).
- [x] Build Stage 1 retrieval using Reciprocal Rank Fusion (RRF) instead of linear weighting.

## Medium Priority
- [x] Write a specific JD intent vector for "hands-on coding" vs "management" to semantically adjust scores.
- [x] Build a robust `reasoning_generator` that outputs exact spec-compliant strings.
- [x] **[L3 FIX]** Implement Adaptive Early Exit Cross-Encoder: probe first 100, exit if high variance, else process all 500.

## Low Priority
- [x] Test the pipeline end-to-end to verify wall-clock time is < 5 minutes on a standard CPU.
- [x] Run the official `validate_submission.py` to ensure output compliance.
- [x] Ensure tie-breaking logic is completely deterministic based on `candidate_id`.

## Senior Engineer Review Fixes (Completed)
- [x] **[L1]** Fix broken multiplicative skill score → Harmonic Mean × log1p(Duration)
- [x] **[L2]** Remove AI penalty vector → Replaced with CPU-cheap Stylometry (TTR + Burstiness) as flag-only
- [x] **[L3]** Adaptive Early Exit Cross-Encoder → average runtime ~90s vs old ~250s
- [x] **[L4]** FAISS index + JD hash for delta invalidation → JD re-encode in 0.5s, not full 30min recompute
- [x] **[L5]** Fix H19 Skill Inflation → max(single_skill_dur) > career*1.2 instead of broken sum check
