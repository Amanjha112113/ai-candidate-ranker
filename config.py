import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get('CANDIDATES_DIR', '/Users/amanjha/Desktop/India_runs_data_and_ai_challenge')

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
CANDIDATES_PATH = os.path.join(DATA_DIR, "candidates.jsonl")

# Create artifacts dir
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

JD_PATH = os.path.join(DATA_DIR, "job_description.docx")

# Artifacts
EMBEDDINGS_PATH = os.path.join(ARTIFACTS_DIR, "embeddings.npy")
CANDIDATE_DATA_PATH = os.path.join(ARTIFACTS_DIR, "candidate_data.pkl")
BM25_INDEX_PATH = os.path.join(ARTIFACTS_DIR, "bm25_index.pkl")
HONEYPOT_RESULTS_PATH = os.path.join(ARTIFACTS_DIR, "honeypot_results.json")
BEHAVIORAL_SCORES_PATH = os.path.join(ARTIFACTS_DIR, "behavioral_scores.json")
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "features.json")

SUBMISSION_PATH = os.path.join(BASE_DIR, "submission.csv")

# Models
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER_MODEL_NAME = "BAAI/bge-reranker-base"

# Constraints & Search settings
TOP_K_STAGE_1 = 500
FINAL_TOP_K = 100
RRF_K = 60

# JD Budget
JD_BUDGET_MIN_LPA = 25
JD_BUDGET_MAX_LPA = 50
