# Problem Statement & True JD Intent (Stored for Offline Reference)

## 1. The Goal
Rank the top 100 candidates from a 100K pool for a **Senior AI Engineer — Founding Team** role at Redrob AI. The ranking must be reproducible locally on CPU within 5 minutes, 16GB RAM, without network calls.

## 2. The True Intent of the JD
The JD is intentionally unconventional and contains explicit traps. It does NOT want a candidate who just has the highest density of AI keywords. 

### What they actually want (The "Shipper"):
- 5-9 years of overall experience (flexible, but 4-5 years applied ML in product companies).
- **Production experience** with embeddings-based retrieval systems (handling drift, index refresh) deployed to real users.
- **Production experience** with vector databases or hybrid search infra (Pinecone, Weaviate, FAISS, etc.).
- Strong Python coding skills (must have written production code in the last 18 months).
- Experience designing evaluation frameworks (NDCG, MAP, A/B testing).
- "Shipper" mentality: willing to ship a working suboptimal system quickly and iterate based on user data.
- Stable tenure: plans to stay for 3+ years.

### Explicit Disqualifiers (The Negative Filters):
1. **Keyword Stuffers**: Candidates with AI keywords but non-engineering titles (e.g., "Marketing Manager").
2. **Pure Research / Academia**: Candidates who have only worked in academic labs or research-only roles without production deployment.
3. **Thin LLM Wrappers**: "AI experience" consists solely of <12 months using LangChain/OpenAI, with no pre-LLM ML production experience.
4. **Non-Coding Architects**: Senior engineers/tech leads who haven't written production code in the past 18 months.
5. **Title Chasers**: Candidates optimizing for titles by switching companies every 1.5 years.
6. **Pure Consulting**: Candidates whose *entire* career is at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini) without product company experience.
7. **Wrong Domain**: Primary expertise is in Computer Vision, Speech, or Robotics without significant NLP/IR exposure.

## 3. The Traps in the Dataset
- **Keyword Stuffers**: Profiles packed with "RAG, Pinecone, Langchain" but with 0 months of experience or irrelevant job titles.
- **Honeypots**: ~80 impossible profiles (e.g., graduated in 2025 but claims 15 years experience, expert in a skill for 0 months). Submissions with >10% honeypots in the top 100 are disqualified.
- **Behavioral Ghosts**: Perfect-on-paper candidates who haven't logged in for 6 months and have a 5% recruiter response rate. These must be severely down-weighted because they are "not actually available."

## 4. Evaluation Metrics
- 50% NDCG@10, 30% NDCG@50, 15% MAP, 5% P@10.
- This means the Top 10 are overwhelmingly the most important. Precision at the very top is everything.
