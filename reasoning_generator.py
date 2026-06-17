def generate_reasoning(candidate, rank, scores):
    """
    Generates the reasoning string in the exact format shown in the sample submission:
    e.g., "HR Manager with 6.1 yrs; 9 AI core skills; response rate 0.76."
    """
    prof = candidate.get("profile", {})
    yoe = float(prof.get("years_of_experience", 0.0))
    title = prof.get("current_title", "Unknown Title").strip()
    
    sig = candidate.get("redrob_signals", {})
    resp_rate = float(sig.get("recruiter_response_rate", 0.0))
    
    skills = candidate.get("skills", [])
    ai_keywords = {
        "python", "machine learning", "deep learning", "nlp", "llm", 
        "embedding", "vector", "pinecone", "weaviate", "milvus", "qdrant", 
        "faiss", "pytorch", "tensorflow", "keras", "transformers", 
        "ranking", "retrieval", "search", "recommendation", "rag", "langchain",
        "artificial intelligence", "data science"
    }
    
    ai_skill_count = 0
    for sk in skills:
        name = sk.get("name", "").lower()
        if any(kw in name for kw in ai_keywords):
            ai_skill_count += 1
            
    reasoning = f"{title} with {yoe:.1f} yrs; {ai_skill_count} AI core skills; response rate {resp_rate:.2f}."
    return reasoning
