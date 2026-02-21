from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text

from app.core.config import settings
from app.core.logger import logger

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass


async def init_db() -> None:
    """Cria as tabelas e ativa a extensão pgvector."""
    async with engine.begin() as conn:
        # Ativa pgvector
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # Importa modelos para que o metadata seja populado
        from app.models.face_embedding import FaceEmbedding  # noqa: F401
        await conn.run_sync(Base.metadata.create_all)
    logger.info("pgvector extension enabled and tables created.")


async def get_db() -> AsyncSession:  # type: ignore[return]
    async with AsyncSessionLocal() as session:
        yield session
