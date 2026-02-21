from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import require_api_key
from app.core.logger import logger
from app.models.schemas import EnrollResponse, DeleteResponse
from app.services.embedding_service import get_face_embedding_service, FaceEmbeddingService
from app.services.face_repository import FaceRepository
from app.utils.image import validate_and_load_image

router = APIRouter()


@router.post(
    "/enroll",
    response_model=EnrollResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Cadastrar rosto",
    description=(
        "Recebe uma imagem e o user_id, extrai o embedding ArcFace "
        "e armazena no banco. Substitui qualquer cadastro anterior do mesmo usuário."
    ),
)
async def enroll_face(
    user_id: str = Form(..., description="ID do usuário no seu sistema de login."),
    image: UploadFile = File(..., description="Foto do rosto (JPEG/PNG/WebP, máx 5 MB)."),
    db: AsyncSession = Depends(get_db),
    _: str = Depends(require_api_key),
    svc: FaceEmbeddingService = Depends(get_face_embedding_service),
):
    image_bytes = await validate_and_load_image(image)

    try:
        embedding = await svc.get_embedding(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    repo = FaceRepository(db)
    face = await repo.upsert(user_id, embedding)

    logger.info("Face enrolled", user_id=user_id, embedding_id=face.id)
    return EnrollResponse(user_id=user_id, embedding_id=face.id)


@router.delete(
    "/enroll/{user_id}",
    response_model=DeleteResponse,
    summary="Remover rosto cadastrado",
    description="Remove todos os embeddings do usuário (ex: ao deletar conta).",
)
async def delete_enrollment(
    user_id: str,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(require_api_key),
):
    repo = FaceRepository(db)
    count = await repo.delete_by_user(user_id)

    logger.info("Face enrollment deleted", user_id=user_id, count=count)
    return DeleteResponse(
        user_id=user_id,
        deleted_count=count,
        message="Embedding removido com sucesso." if count > 0 else "Nenhum embedding encontrado.",
    )
