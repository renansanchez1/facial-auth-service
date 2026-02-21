import asyncio
import io
from functools import lru_cache
from typing import List

import numpy as np
from PIL import Image

from app.core.config import settings
from app.core.logger import logger


class FaceEmbeddingService:
    """
    Wrapper em torno do InsightFace (modelo ArcFace buffalo_l).
    Carregado uma única vez (singleton) por processo.
    """

    def __init__(self) -> None:
        self._app = None
        self._lock = asyncio.Lock()

    def _load_model(self):
        """Carrega o modelo de forma lazy (apenas na primeira chamada)."""
        if self._app is not None:
            return

        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                "insightface não instalado. Execute: pip install insightface"
            ) from exc

        logger.info("Loading InsightFace model...", model=settings.INSIGHTFACE_MODEL)
        self._app = FaceAnalysis(
            name=settings.INSIGHTFACE_MODEL,
            allowed_modules=["detection", "recognition"],
        )
        self._app.prepare(ctx_id=settings.INSIGHTFACE_CTX_ID, det_size=(640, 640))
        logger.info("InsightFace model loaded.")

    async def get_embedding(self, image_bytes: bytes) -> List[float]:
        """
        Detecta o rosto e retorna o embedding ArcFace normalizado (L2).

        Raises:
            ValueError: se nenhum ou mais de um rosto for detectado.
        """
        async with self._lock:
            # Carrega o modelo na primeira chamada (thread-safe via lock)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_model)

        # Converte bytes → numpy array (BGR para InsightFace)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img)[:, :, ::-1]  # RGB → BGR

        # Inferência em executor para não bloquear o event loop
        loop = asyncio.get_event_loop()
        faces = await loop.run_in_executor(None, self._app.get, img_np)

        if len(faces) == 0:
            raise ValueError("Nenhum rosto detectado na imagem.")
        if len(faces) > 1:
            raise ValueError(
                f"{len(faces)} rostos detectados. Envie uma imagem com apenas um rosto."
            )

        face = faces[0]
        embedding: np.ndarray = face.normed_embedding  # já normalizado L2 pelo InsightFace
        return embedding.tolist()


@lru_cache(maxsize=1)
def get_face_embedding_service() -> FaceEmbeddingService:
    return FaceEmbeddingService()
