import uuid
from datetime import datetime, timezone

from sqlalchemy import String, DateTime, Index, text
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from app.core.config import settings
from app.core.database import Base


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    # O embedding não é armazenado em texto — pgvector guarda como vetor binário eficiente.
    embedding: Mapped[list] = mapped_column(
        Vector(settings.EMBEDDING_SIZE), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        # Índice HNSW para busca por similaridade eficiente (ANN)
        Index(
            "ix_face_embeddings_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    def __repr__(self) -> str:
        return f"<FaceEmbedding id={self.id} user_id={self.user_id}>"
