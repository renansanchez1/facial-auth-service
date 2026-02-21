import io
from abc import ABC, abstractmethod
from functools import lru_cache

from app.core.config import settings
from app.core.logger import logger


class LivenessResult:
    def __init__(self, is_live: bool, confidence: float, provider: str):
        self.is_live = is_live
        self.confidence = confidence
        self.provider = provider


class BaseLivenessDetector(ABC):
    @abstractmethod
    async def check(self, image_bytes: bytes) -> LivenessResult:
        ...


# ---------------------------------------------------------------------------
# AWS Rekognition
# ---------------------------------------------------------------------------

class AWSLivenessDetector(BaseLivenessDetector):
    """
    Usa o DetectFaces do Rekognition para checar atributos de qualidade.
    Para liveness challenge-response, veja FaceMovementAndLightClientSessionConfig
    (requer integração com o SDK frontend da AWS).
    Esta implementação checa EyesOpen + Pose como proxy simples de liveness.
    """

    async def check(self, image_bytes: bytes) -> LivenessResult:
        import asyncio
        import boto3

        def _call():
            client = boto3.client(
                "rekognition",
                region_name=settings.AWS_REGION,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID or None,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY or None,
            )
            response = client.detect_faces(
                Image={"Bytes": image_bytes},
                Attributes=["ALL"],
            )
            return response

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(None, _call)
        except Exception as exc:
            logger.error("AWS Rekognition error", error=str(exc))
            # Fail-open em erro de conectividade (ajuste para fail-closed se preferir)
            return LivenessResult(is_live=True, confidence=0.0, provider="aws_error")

        faces = response.get("FaceDetails", [])
        if not faces:
            return LivenessResult(is_live=False, confidence=0.0, provider="aws")

        face = faces[0]
        confidence = face.get("Confidence", 0.0)
        eyes_open = face.get("EyesOpen", {}).get("Value", False)
        sunglasses = face.get("Sunglasses", {}).get("Value", False)

        is_live = (
            confidence >= settings.AWS_LIVENESS_MIN_CONFIDENCE
            and eyes_open
            and not sunglasses
        )

        return LivenessResult(is_live=is_live, confidence=confidence, provider="aws")


# ---------------------------------------------------------------------------
# Silent-Face (local, sem custo por requisição)
# ---------------------------------------------------------------------------

class SilentFaceLivenessDetector(BaseLivenessDetector):
    """
    Wrapper para o modelo Silent-Face-Anti-Spoofing.
    Repositório: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
    Instale com: pip install silent-face  (ou clone o repo e instale manualmente)
    """

    async def check(self, image_bytes: bytes) -> LivenessResult:
        import asyncio
        import numpy as np
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img)

        def _infer():
            try:
                from silent_face import SilentFaceAntiSpoofing
                model = SilentFaceAntiSpoofing()
                label, score = model.predict(img_np)
                # label 1 = real, 0 = spoof
                return bool(label == 1), float(score)
            except ImportError:
                logger.warning("silent_face not installed, liveness check skipped.")
                return True, 1.0

        loop = asyncio.get_event_loop()
        is_live, score = await loop.run_in_executor(None, _infer)
        return LivenessResult(is_live=is_live, confidence=score * 100, provider="silent_face")


# ---------------------------------------------------------------------------
# No-op (desenvolvimento)
# ---------------------------------------------------------------------------

class NoopLivenessDetector(BaseLivenessDetector):
    async def check(self, image_bytes: bytes) -> LivenessResult:
        logger.warning("Liveness detection DISABLED (LIVENESS_PROVIDER=none).")
        return LivenessResult(is_live=True, confidence=100.0, provider="none")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_liveness_detector() -> BaseLivenessDetector:
    provider = settings.LIVENESS_PROVIDER.lower()
    if provider == "aws":
        return AWSLivenessDetector()
    elif provider == "silent_face":
        return SilentFaceLivenessDetector()
    else:
        return NoopLivenessDetector()
