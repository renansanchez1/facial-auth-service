from typing import List, Optional, Tuple
from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.face_embedding import FaceEmbedding
from app.core.config import settings


class FaceRepository:

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def upsert(self, user_id: str, embedding: List[float]) -> FaceEmbedding:
        """
        Cria ou substitui o embedding do usuário.
        Cada usuário tem apenas um embedding ativo (simplifica revogação).
        """
        await self.delete_by_user(user_id)

        face = FaceEmbedding(user_id=user_id, embedding=embedding)
        self.db.add(face)
        await self.db.commit()
        await self.db.refresh(face)
        return face

    async def find_closest(
        self, user_id: str, query_embedding: List[float]
    ) -> Optional[Tuple[FaceEmbedding, float]]:
        """
        Busca o embedding mais próximo para o user_id informado.
        Retorna (FaceEmbedding, distância_cosine) ou None se não encontrado.
        A distância cosine varia de 0 (idênticos) a 2 (opostos).
        """
        # pgvector: <=> = distância cosine
        stmt = (
            select(
                FaceEmbedding,
                FaceEmbedding.embedding.cosine_distance(query_embedding).label("distance"),
            )
            .where(FaceEmbedding.user_id == user_id)
            .order_by("distance")
            .limit(1)
        )
        result = await self.db.execute(stmt)
        row = result.first()
        if row is None:
            return None
        return row[0], float(row[1])

    async def delete_by_user(self, user_id: str) -> int:
        stmt = delete(FaceEmbedding).where(FaceEmbedding.user_id == user_id)
        result = await self.db.execute(stmt)
        await self.db.commit()
        return result.rowcount

    async def exists(self, user_id: str) -> bool:
        stmt = select(func.count()).where(FaceEmbedding.user_id == user_id)
        result = await self.db.execute(stmt)
        return result.scalar_one() > 0
