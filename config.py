import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Detect Google Colab (works in notebook kernel and subprocesses like !python)
IS_COLAB = 'google.colab' in sys.modules or 'COLAB_RELEASE_TAG' in os.environ or os.path.exists('/content/drive')

if IS_COLAB:
    DEFAULT_DATA_DIR = '/content/drive/MyDrive/India_runs_data_and_ai_challenge'
else:
    DEFAULT_DATA_DIR = '/Users/amanjha/Desktop/India_runs_data_and_ai_challenge'

DATA_DIR = os.environ.get('CANDIDATES_DIR', DEFAULT_DATA_DIR)

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
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # Reverted to bge-small (33M, 384-dim) for speed and memory efficiency
                                                    # Higher semantic accuracy; only affects precompute.py (offline).
CROSS_ENCODER_MODEL_NAME = "BAAI/bge-reranker-base"

# Constraints & Search settings
TOP_K_STAGE_1 = 500
FINAL_TOP_K = 100
RRF_K = 60

# JD Budget
JD_BUDGET_MIN_LPA = 25
JD_BUDGET_MAX_LPA = 50
