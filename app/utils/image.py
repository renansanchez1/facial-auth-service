import io
from fastapi import UploadFile, HTTPException, status
from PIL import Image

from app.core.config import settings


async def validate_and_load_image(file: UploadFile) -> bytes:
    """
    Valida content-type, tamanho e integridade da imagem.
    Retorna os bytes brutos validados.
    """
    # 1. Content-type
    if file.content_type not in settings.ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Tipo de imagem não suportado: {file.content_type}. Use JPEG, PNG ou WebP.",
        )

    # 2. Leitura e tamanho
    contents = await file.read()
    max_bytes = int(settings.MAX_IMAGE_SIZE_MB * 1024 * 1024)
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Imagem maior que {settings.MAX_IMAGE_SIZE_MB} MB.",
        )

    # 3. Integridade (PIL consegue abrir?)
    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Arquivo de imagem inválido ou corrompido.",
        )

    return contents
