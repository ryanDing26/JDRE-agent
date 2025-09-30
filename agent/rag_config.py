import os

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Paste your OpenAI key between the quotes (please don't - Ryan):
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# Chat model (OpenAI) used to synthesize & refine answers
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# One or more SentenceTransformer embedding models (comma-separated or single)
# Good biomedical options:
#  - "pritamdeka/S-BioBert-snli-multinli-stsb"
#  - "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
#  - "sentence-transformers/all-MiniLM-L6-v2" (fast baseline)
EMBED_MODEL_NAMES = os.getenv(
    "EMBED_MODEL_NAMES",
    "pritamdeka/S-BioBert-snli-multinli-stsb,cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
)

# Use cosine similarity (recommended for normalized embeddings)
USE_COSINE = True

# FAISS GPU usage (pip-only setup often lacks faiss-gpu; default False)
FAISS_USE_GPU = os.getenv("FAISS_USE_GPU", "0") == "1"
FAISS_GPU_ID = int(os.getenv("FAISS_GPU_ID", "0"))

# Chunking defaults
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))      # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200")) # overlapping characters

# Retrieval knobs
# TOP_K = int(os.getenv("TOP_K", "12"))                  # candidates per model from FAISS
# FINAL_K = int(os.getenv("FINAL_K", "5"))               # final snippets after re-rank/MMR
# EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "96"))
TOP_K = 12
FINAL_K = 5
EMBED_BATCH_SIZE = 96 # TODO make it based on available number of workers

# Hybrid scoring weights
# ALPHA_EMBED = float(os.getenv("ALPHA_EMBED", "0.72"))   # embedding similarity weight
# BETA_KEYWORD = float(os.getenv("BETA_KEYWORD", "0.23")) # keyword/phrase overlap weight
# GAMMA_PHRASE = float(os.getenv("GAMMA_PHRASE", "0.12")) # exact phrase bonus
# DELTA_OFFTOPIC = float(os.getenv("DELTA_OFFTOPIC", "0.25")) # off-topic penalty

ALPHA_EMBED = 0.72
BETA_KEYWORD = 0.23
GAMMA_PHRASE = 0.12
DELTA_OFFTOPIC = 0.25

# MMR
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.7"))     # 1.0 = relevance only, 0.0 = diversity
MMR_LAMBDA = 0.7

# Index cap (safety for accidental huge loads)
MAX_CHUNKS_WARN = int(os.getenv("MAX_CHUNKS_WARN", "300000"))