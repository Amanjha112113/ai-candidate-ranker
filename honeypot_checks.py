import datetime
import json

REFERENCE_DATE = datetime.date(2026, 6, 1)

def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.date.fromisoformat(date_str[:10])
    except ValueError:
        return None

def calculate_actual_months(start_date_str, end_date_str, is_current):
    start = parse_date(start_date_str)
    if not start:
        return 0
    if is_current or not end_date_str:
        end = REFERENCE_DATE
    else:
        end = parse_date(end_date_str)
        if not end:
            return 0
    
    if end < start:
        return -1 # Invalid, handled by H04
    
    return (end.year - start.year) * 12 + (end.month - start.month)

def run_honeypot_checks(candidate, global_skill_freq):
    """
    Runs the 16 calibrated checks from the PDF + new checks (H17, H18).
    Returns: {"is_hard_honeypot": bool, "flags": list, "penalty": float, "soft_flags": list, "hard": bool}
    """
    flags = []
    hard_flag = False
    
    prof = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    edu = candidate.get("education", [])
    skills = candidate.get("skills", [])
    sig = candidate.get("redrob_signals", {})
    
    # Pre-computations
    years_of_experience = prof.get("years_of_experience", 0)
    
    grad_year = 2025
    if edu:
        grad_years = [e.get("end_year") for e in edu if e.get("end_year")]
        if grad_years:
            grad_year = max(grad_years)
            
    total_career_months = 0
    
    # H04: End date before start date
    for job in career:
        sd = parse_date(job.get("start_date"))
        ed = parse_date(job.get("end_date"))
        if sd and ed and ed < sd:
            flags.append("H04")
            hard_flag = True
            break
            
    # H01: Career duration math mismatch
    for job in career:
        stated = job.get("duration_months", 0)
        actual = calculate_actual_months(job.get("start_date"), job.get("end_date"), job.get("is_current"))
        total_career_months += stated
        if actual >= 0 and abs(actual - stated) > 6:
            flags.append("H01")
            break
            
    # H02: Post-graduation experience impossible
    if years_of_experience > (2025 - grad_year) + 8:
        flags.append("H02")
        
    # H03: Overlapping Full-Time Careers (>60 days)
    if len(career) > 1:
        for i in range(len(career)):
            for j in range(i+1, len(career)):
                s1 = parse_date(career[i].get('start_date'))
                e1 = parse_date(career[i].get('end_date')) if not career[i].get('is_current') else REFERENCE_DATE
                s2 = parse_date(career[j].get('start_date'))
                e2 = parse_date(career[j].get('end_date')) if not career[j].get('is_current') else REFERENCE_DATE
                
                if s1 and e1 and s2 and e2:
                    # Check overlap
                    latest_start = max(s1, s2)
                    earliest_end = min(e1, e2)
                    if latest_start < earliest_end:
                        overlap_days = (earliest_end - latest_start).days
                        if overlap_days > 60:
                            flags.append("H03")
                            break
            if "H03" in flags:
                break

    # H10: YoE vs Career Duration Sum gap > 5
    total_career_months_actual = 0
    for job in career:
        actual = calculate_actual_months(job.get('start_date'), job.get('end_date'), job.get('is_current'))
        if actual > 0:
            total_career_months_actual += actual
            
    if abs(years_of_experience - (total_career_months_actual / 12.0)) > 5:
        flags.append("H10")

    total_skill_months = 0
    expert_skills = 0
    total_endorsements = 0
    
    for sk in skills:
        dur = sk.get("duration_months", 0)
        total_skill_months += dur
        prof_level = sk.get("proficiency", "").lower()
        endorse = sk.get("endorsements", 0)
        total_endorsements += endorse
        
        # H05: Single skill > entire career
        if dur > (2025 - grad_year) * 12 + 12:
            flags.append("H05")
            hard_flag = True
            
        # H08: Expert/Advanced with 0 months
        if prof_level in ["expert", "advanced"] and dur == 0:
            flags.append("H08")
            hard_flag = True
            
        if prof_level == "expert":
            expert_skills += 1

    # H11: Total Skill Months > 10x Career Months
    if total_skill_months > total_career_months * 10:
        flags.append("H11")
        
    # H15: 3+ expert skills but zero endorsements on all skills
    if expert_skills >= 3 and total_endorsements == 0:
        flags.append("H15")
        
    # H06: Signup Date After Last Active Date (>90 days)
    su = parse_date(sig.get("signup_date"))
    la = parse_date(sig.get("last_active_date"))
    if su and la and (su - la).days > 90:
        flags.append("H06")
        hard_flag = True
        
    # H13: Offers accepted, 0 interviews
    if sig.get("offer_acceptance_rate", -1) > 0 and sig.get("interview_completion_rate", 1) == 0:
        flags.append("H13")
        hard_flag = True
        
    # H14: High response rate but > 1 week response time
    if sig.get("recruiter_response_rate", 0) > 0.8 and sig.get("avg_response_time_hours", 0) > 168:
        flags.append("H14")
        
    # H09: Education Start Year >= End Year
    for ed in edu:
        sy = ed.get("start_year")
        ey = ed.get("end_year")
        if sy and ey and sy >= ey:
            flags.append("H09")
            hard_flag = True
            break
            
    # H12: GitHub > 70, no coding skills
    if sig.get("github_activity_score", -1) > 70:
        prog_langs = {"python", "java", "c++", "c#", "javascript", "typescript", "go", "ruby", "rust", "php", "swift", "kotlin", "r"}
        has_prog = any(sk.get("name", "").lower() in prog_langs for sk in skills)
        if not has_prog:
            flags.append("H12")
            
    # H16: Copy-pasted job descriptions
    if len(career) >= 2:
        descs = [job.get("description", "")[:100] for job in career if len(job.get("description", "")) >= 100]
        if len(descs) != len(set(descs)):
            flags.append("H16")
            
    # H19: Skill Inflation Paradox
    if total_career_months > 0:
        max_skill_dur = max((sk.get("duration_months", 0) for sk in skills), default=0)
        if max_skill_dur > total_career_months * 1.2:
            flags.append("H19")

    # NEW: H17 Skill-Job Mismatch
    # Removed as per feedback to eliminate false positives
    
    # NEW: H18 Rare Skill Combo
    rare_skills = 0
    for sk in skills:
        freq = global_skill_freq.get(sk.get("name", "").lower(), 0) / 100000.0
        if freq < 0.01 and sk.get("proficiency", "").lower() in ["advanced", "expert"]:
            rare_skills += 1
    if rare_skills >= 3:
        flags.append("H18_rare")

    # NEW: H07 Salary Inversion (Updated logic)
    sal = sig.get("expected_salary_range_inr_lpa", {})
    if sal.get("min", 0) > sal.get("max", 0):
        # Check for OTHER penalizing flags (excluding H07-related)
        other_penalizing = [f for f in flags if f not in ["H07", "H07_INFO"] and not f.endswith("_INFO")]
        if len(other_penalizing) > 0:
            flags.append("H07")
        else:
            flags.append("H07_INFO")
            
    # Filter soft flags vs hard flags
    soft_flags = [f for f in flags if not f.endswith("_INFO") and f not in ["H04", "H05", "H06", "H08", "H09", "H13"]]

    # Multiplicative Credibility Strategy (Optional, but we return structured data now)
    penalty = 1.0
    if hard_flag:
        penalty = 0.0
    else:
        soft_multipliers = {
            "H01": 0.95, "H02": 0.85, "H03": 0.85, "H07": 0.50, "H10": 0.80,
            "H11": 0.75, "H12": 0.85, "H14": 0.90, "H15": 0.90, "H16": 0.95,
            "H18_rare": 0.80, "H19": 0.85
        }
        for f in flags:
            penalty *= soft_multipliers.get(f, 1.0)
            
    return {
        "hard": hard_flag,
        "is_hard_honeypot": hard_flag,
        "flags": flags,
        "soft_flags": soft_flags,
        "penalty": penalty
    }
