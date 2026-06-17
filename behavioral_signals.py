import datetime

def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.date.fromisoformat(date_str[:10])
    except ValueError:
        return None

def compute_behavioral_signals(candidate):
    sig = candidate.get("redrob_signals", {})
    prof = candidate.get("profile", {})
    ref_date = datetime.date(2026, 6, 1)
    
    # 1. Profile Completeness
    completeness = sig.get("profile_completeness_score", 0) / 100.0
    
    # 2. Signup Date
    su = parse_date(sig.get("signup_date"))
    if su:
        days = (ref_date - su).days
        platform_tenure = min((days / 365.0) / 3.0, 1.0)
    else:
        platform_tenure = 0.5
        
    # 3. Last Active
    la = parse_date(sig.get("last_active_date"))
    recency = 0.2
    if la:
        days_ago = (ref_date - la).days
        if days_ago <= 7: recency = 1.0
        elif days_ago <= 30: recency = 0.9
        elif days_ago <= 90: recency = 0.7
        elif days_ago <= 180: recency = 0.4
        
    # 4. Open to work
    open_to_work = 1.0 if sig.get("open_to_work_flag") else 0.3
    
    # 5. Profile views
    views = min(sig.get("profile_views_received_30d", 0) / 100.0, 1.0)
    
    # 6. Applications
    apps = sig.get("applications_submitted_30d", 0)
    if apps == 0: apps_norm = 0.2
    elif apps <= 5: apps_norm = 0.8
    elif apps <= 10: apps_norm = 1.0
    else: apps_norm = 0.6
    
    # 7. Response rate
    response_rate = sig.get("recruiter_response_rate", 0)
    
    # 8. Response time
    hours = sig.get("avg_response_time_hours", 0)
    import math
    time_norm = math.exp(-hours / 72.0)
    
    # 9. Assessments
    assess = sig.get("skill_assessment_scores", {})
    if not assess:
        assess_norm = 0.3
    else:
        avg_score = sum(assess.values()) / len(assess)
        assess_norm = 0.3 + 0.7 * (avg_score / 100.0)
        
    # 10. Connections
    connections = min(sig.get("connection_count", 0) / 500.0, 1.0)
    
    # 11. Endorsements
    endorsements = min(sig.get("endorsements_received", 0) / 50.0, 1.0)
    
    # 12. Notice Period
    notice = sig.get("notice_period_days", 0)
    if notice <= 30: notice_norm = 1.0
    elif notice <= 60: notice_norm = 0.8
    elif notice <= 90: notice_norm = 0.6
    else: notice_norm = 0.4
    
    # Logistics Multiplier (Work Mode & Relocation only)
    mode = sig.get("preferred_work_mode", "flexible").lower()
    if mode == "hybrid": mode_score = 1.0
    elif mode == "flexible": mode_score = 0.9
    elif mode == "remote": mode_score = 0.8
    else: mode_score = 0.8
    
    loc = prof.get("location", "").lower()
    willing = sig.get("willing_to_relocate", False)
    if "pune" in loc or "noida" in loc or "delhi" in loc or "mumbai" in loc or "hyderabad" in loc or "ncr" in loc:
        relocation_score = 1.0
    else:
        relocation_score = 0.7 if willing else 0.5
        
    jd_fit_multiplier = 0.6 * relocation_score + 0.4 * mode_score
        
    # 16. GitHub
    gh = sig.get("github_activity_score", -1)
    gh_norm = 0.5 if gh < 0 else 0.5 + 0.5 * (gh / 100.0)
    
    # 17. Search Appearance
    search = min(sig.get("search_appearance_30d", 0) / 100.0, 1.0)
    
    # 18. Saved by Recruiter
    saved = min(sig.get("saved_by_recruiters_30d", 0) / 20.0, 1.0)
    
    # 19. Interview completion
    interview = sig.get("interview_completion_rate", 0)
    
    # 20. Offer acceptance
    offer = sig.get("offer_acceptance_rate", -1)
    offer_norm = 0.5 if offer < 0 else 0.3 + 0.7 * offer
    
    # 21-23. Verified
    email = 1.0 if sig.get("verified_email") else 0.3
    phone = 1.0 if sig.get("verified_phone") else 0.3
    linkedin = 1.0 if sig.get("linkedin_connected") else 0.3

    # Composites
    availability = (
        0.25 * open_to_work +
        0.25 * recency +
        0.25 * response_rate +
        0.10 * time_norm +
        0.10 * notice_norm +
        0.05 * apps_norm
    )
    
    market_validation = (
        0.35 * saved +
        0.25 * search +
        0.25 * views +
        0.15 * endorsements
    )
    
    trust = (
        0.30 * completeness +
        0.25 * email +
        0.25 * phone +
        0.20 * linkedin
    )
    
    reliability = (
        0.50 * interview +
        0.30 * offer_norm +
        0.20 * connections
    )
    
    technical = (
        0.50 * assess_norm +
        0.30 * gh_norm +
        0.20 * platform_tenure
    )
    
    weighted_composite = (
        0.30 * availability +
        0.25 * market_validation +
        0.15 * trust +
        0.15 * reliability +
        0.15 * technical
    )

    open_to_work = sig.get('open_to_work_flag', False)
    la_date = parse_date(sig.get('last_active_date'))
    last_active_days = (ref_date - la_date).days if la_date else 365

    if not open_to_work and last_active_days > 180:
        # Harsher penalty for truly inactive
        behavioral_multiplier = 0.1 + 0.9 * weighted_composite
    else:
        behavioral_multiplier = 0.3 + 0.7 * weighted_composite
    

    return {
        "behavioral_multiplier": behavioral_multiplier,
        "jd_fit_multiplier": jd_fit_multiplier,
        "composites": {
            "availability": availability,
            "market_validation": market_validation,
            "trust": trust,
            "reliability": reliability,
            "technical": technical
        }
    }
