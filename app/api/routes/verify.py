from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.security import require_api_key
from app.core.logger import logger
from app.models.schemas import VerifyResponse
from app.services.embedding_service import get_face_embedding_service, FaceEmbeddingService
from app.services.face_repository import FaceRepository
from app.services.liveness_service import get_liveness_detector, BaseLivenessDetector
from app.utils.image import validate_and_load_image

router = APIRouter()


@router.post(
    "/verify",
    response_model=VerifyResponse,
    summary="Verificar rosto",
    description=(
        "Recebe uma imagem ao vivo e o user_id. "
        "Realiza detecção de vivacidade (liveness) e compara o embedding "
        "com o cadastrado. Retorna match + score de confiança."
    ),
)
async def verify_face(
    user_id: str = Form(...),
    image: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    _: str = Depends(require_api_key),
    embed_svc: FaceEmbeddingService = Depends(get_face_embedding_service),
    liveness_svc: BaseLivenessDetector = Depends(get_liveness_detector),
):
    image_bytes = await validate_and_load_image(image)

    # 1. Verifica se o usuário tem rosto cadastrado
    repo = FaceRepository(db)
    if not await repo.exists(user_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Usuário '{user_id}' não possui rosto cadastrado.",
        )

    # 2. Liveness detection (executa em paralelo com a extração de embedding)
    import asyncio
    liveness_result, query_embedding = await asyncio.gather(
        liveness_svc.check(image_bytes),
        _extract_embedding(embed_svc, image_bytes),
    )

    if isinstance(query_embedding, ValueError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(query_embedding),
        )

    # 3. Liveness falhou
    if not liveness_result.is_live:
        logger.warning(
            "Liveness check failed",
            user_id=user_id,
            provider=liveness_result.provider,
            confidence=liveness_result.confidence,
        )
        return VerifyResponse(
            user_id=user_id,
            match=False,
            confidence=0.0,
            liveness_passed=False,
            message="Liveness check falhou. Use uma imagem ao vivo.",
        )

    # 4. Comparação de embeddings
    result = await repo.find_closest(user_id, query_embedding)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Embedding não encontrado.")

    _, distance = result
    # Converte distância cosine (0–2) em similarity (0–1)
    similarity = 1.0 - (distance / 2.0)
    match = distance <= settings.SIMILARITY_THRESHOLD

    logger.info(
        "Face verification",
        user_id=user_id,
        match=match,
        distance=round(distance, 4),
        similarity=round(similarity, 4),
        liveness_provider=liveness_result.provider,
    )

    return VerifyResponse(
        user_id=user_id,
        match=match,
        confidence=round(similarity, 4),
        liveness_passed=True,
        message="Identidade verificada com sucesso." if match else "Rosto não reconhecido.",
    )


async def _extract_embedding(svc: FaceEmbeddingService, image_bytes: bytes):
    """Helper que captura ValueError para uso com asyncio.gather."""
    try:
        return await svc.get_embedding(image_bytes)
    except ValueError as exc:
        return exc
