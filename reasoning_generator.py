def generate_reasoning(candidate, rank, scores):
    prof = candidate.get("profile", {})
    yoe = float(prof.get("years_of_experience", 0.0))
    title = prof.get("current_title", "Unknown").strip()
    
    sig = candidate.get("redrob_signals", {})
    resp_rate = float(sig.get("recruiter_response_rate", 0.0))
    
    # Check for production/retrieval keywords in latest roles
    prod_words = ['production', 'shipped', 'deployed', 'scaled', 'real-world']
    retrieval_words = ['retrieval', 'ranking', 'search', 'recommendation', 'embedding', 'vector', 'hybrid']
    
    has_prod = False
    has_retrieval = False
    company = prof.get("current_company", "a product company")
    if not company:
        company = "a product company"
    
    for job in candidate.get('career_history', []):
        desc = job.get('description', '').lower()
        if any(w in desc for w in prod_words): has_prod = True
        if any(w in desc for w in retrieval_words): has_retrieval = True
        
    tech_str = ""
    if has_prod and has_retrieval:
        tech_str = f"Shipped production retrieval/ranking systems at {company}."
    elif has_prod:
        tech_str = f"Strong production ML background at {company}."
    elif has_retrieval:
        tech_str = f"Deep expertise in search/retrieval systems."
    else:
        tech_str = f"Solid ML/AI foundation."
        
    # Behavioral fact
    beh_str = f"Highly active ({int(resp_rate*100)}% response rate)."
    if resp_rate < 0.5:
        beh_str = "Strong JD match compensates for moderate response rate."
        
    reasoning = f"{tech_str} {title} with {yoe:.1f}yr exp. {beh_str}"
    
    # Truncate to ensure concise
    if len(reasoning) > 150:
        reasoning = reasoning[:147] + "..."
        
    return reasoning
