"""Application settings — reads from environment variables only. No side effects."""
import os

# Vector DB connection
VECTOR_DB_NAME: str = os.environ.get("VECTOR_DB_NAME", "")
VECTOR_DB_USER: str = os.environ.get("VECTOR_DB_USER", "")
VECTOR_DB_PASSWORD: str = os.environ.get("VECTOR_DB_PASSWORD", "")
VECTOR_DB_HOST: str = os.environ.get("VECTOR_DB_HOST", "localhost")
VECTOR_DB_PORT: int = int(os.environ.get("VECTOR_DB_PORT", "5432"))
VECTOR_DB_SCHEMA: str = os.environ.get("VECTOR_DB_SCHEMA", "vectors")
VECTOR_DB_ARTICLES_TABLE: str = os.environ.get("VECTOR_DB_ARTICLES_TABLE", "articles")
VECTOR_DB_CHUNKS_TABLE: str = os.environ.get("VECTOR_DB_CHUNKS_TABLE", "article_chunks")

# LLM — Gemini API key (shared across all models)
RAG_GEMINI_API_KEY: str = os.environ.get("RAG_GEMINI_API_KEY", "")

# LLM provider chain — priority 1 is tried first; falls back on rate-limit or failure
GEMINI_MODEL_1: str = os.environ.get("GEMINI_MODEL_1", "gemini-3-flash-preview")
GEMINI_RPM_1: int = int(os.environ.get("GEMINI_RPM_1", "5"))
GEMINI_TPM_1: int = int(os.environ.get("GEMINI_TPM_1", "250000"))
GEMINI_RPD_1: int = int(os.environ.get("GEMINI_RPD_1", "20"))

GEMINI_MODEL_2: str = os.environ.get("GEMINI_MODEL_2", "gemini-3.1-flash-lite")
GEMINI_RPM_2: int = int(os.environ.get("GEMINI_RPM_2", "15"))
GEMINI_TPM_2: int = int(os.environ.get("GEMINI_TPM_2", "250000"))
GEMINI_RPD_2: int = int(os.environ.get("GEMINI_RPD_2", "500"))

GEMINI_MODEL_3: str = os.environ.get("GEMINI_MODEL_3", "gemini-2.5-flash")
GEMINI_RPM_3: int = int(os.environ.get("GEMINI_RPM_3", "5"))
GEMINI_TPM_3: int = int(os.environ.get("GEMINI_TPM_3", "250000"))
GEMINI_RPD_3: int = int(os.environ.get("GEMINI_RPD_3", "20"))

GEMINI_MODEL_4: str = os.environ.get("GEMINI_MODEL_4", "gemini-2.5-flash-lite")
GEMINI_RPM_4: int = int(os.environ.get("GEMINI_RPM_4", "10"))
GEMINI_TPM_4: int = int(os.environ.get("GEMINI_TPM_4", "250000"))
GEMINI_RPD_4: int = int(os.environ.get("GEMINI_RPD_4", "20"))

# Dense embedding (Gemini API)
RAG_DENSE_MODEL: str = os.environ.get("RAG_DENSE_MODEL", "gemini-embedding-001")
RAG_DENSE_DIMENSION: int = int(os.environ.get("RAG_DENSE_DIMENSION", "768"))

# Sparse embedding (external fastembed service)
RAG_SPARSE_ENDPOINT_URL: str = os.environ.get("RAG_SPARSE_ENDPOINT_URL", "http://fastembed:8080")
RAG_SPARSE_DIMENSION: int = int(os.environ.get("RAG_SPARSE_DIMENSION", "30522"))

# Reranker (in-process fastembed)
RAG_RERANKER_MODEL: str = os.environ.get("RAG_RERANKER_MODEL", "")

# Chat tuning
CHATBOT_MAX_CONTEXT_CHUNKS: int = int(os.environ.get("CHATBOT_MAX_CONTEXT_CHUNKS", "10"))
CHATBOT_MAX_TOKENS: int = int(os.environ.get("CHATBOT_MAX_TOKENS", "2048"))
CHATBOT_RETRIEVAL_MIN_SCORE: float = float(os.environ.get("CHATBOT_RETRIEVAL_MIN_SCORE", "0.0"))
CHATBOT_RERANKER_MIN_SCORE: float = float(os.environ.get("CHATBOT_RERANKER_MIN_SCORE", "0.0"))

# Auth
CHAT_SERVICE_API_KEY: str = os.environ.get("CHAT_SERVICE_API_KEY", "")

# Observability
APP_ENV: str = os.environ.get("APP_ENV", "local")
GRAFANA_LOKI_URL: str = os.environ.get("GRAFANA_LOKI_URL", "")
GRAFANA_LOKI_USER: str = os.environ.get("GRAFANA_LOKI_USER", "")
GRAFANA_API_KEY: str = os.environ.get("GRAFANA_API_KEY", "")
