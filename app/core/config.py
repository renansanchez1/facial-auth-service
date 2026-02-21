from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── App ──────────────────────────────────────────────────────────────────
    ENV: str = "development"
    SECRET_KEY: str = "change-me-in-production"

    # ── Database (PostgreSQL + pgvector) ─────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost:5432/facial_auth"

    # ── InsightFace ───────────────────────────────────────────────────────────
    INSIGHTFACE_MODEL: str = "buffalo_l"          # buffalo_l = ArcFace R100 (mais preciso)
    INSIGHTFACE_CTX_ID: int = -1                  # -1 = CPU; 0 = primeira GPU
    EMBEDDING_SIZE: int = 512
    SIMILARITY_THRESHOLD: float = 0.40            # distância cosine (menor = mais similar)

    # ── Liveness (AWS Rekognition) ────────────────────────────────────────────
    LIVENESS_PROVIDER: str = "aws"                # "aws" | "silent_face" | "none"
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_LIVENESS_MIN_CONFIDENCE: float = 90.0

    # ── Rate limiting ─────────────────────────────────────────────────────────
    RATE_LIMIT_REQUESTS: int = 10
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    # ── Security ──────────────────────────────────────────────────────────────
    API_KEY_HEADER: str = "X-API-Key"
    API_KEYS: List[str] = []                      # lista de chaves válidas

    # ── CORS / Hosts ──────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = ["*"]
    ALLOWED_HOSTS: List[str] = ["*"]

    # ── Imagens ───────────────────────────────────────────────────────────────
    MAX_IMAGE_SIZE_MB: float = 5.0
    ALLOWED_CONTENT_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp"]


settings = Settings()
