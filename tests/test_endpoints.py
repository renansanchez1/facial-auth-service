"""
Testes de integração para os endpoints de enrollment e verificação.
Execute com: pytest tests/ -v
"""

import io
import numpy as np
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch, MagicMock

from app.main import app
from app.core.config import settings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


def _fake_image() -> bytes:
    """Gera uma imagem PNG mínima válida para os testes."""
    from PIL import Image
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _fake_embedding() -> list:
    vec = np.random.randn(settings.EMBEDDING_SIZE).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health(client: AsyncClient):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Enrollment
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enroll_success(client: AsyncClient):
    emb = _fake_embedding()

    with (
        patch("app.api.routes.enroll.get_face_embedding_service") as mock_svc_factory,
        patch("app.api.routes.enroll.FaceRepository") as MockRepo,
    ):
        mock_svc = AsyncMock()
        mock_svc.get_embedding = AsyncMock(return_value=emb)
        mock_svc_factory.return_value = mock_svc

        mock_repo = AsyncMock()
        mock_repo.upsert.return_value = MagicMock(id="abc-123", user_id="user_1")
        MockRepo.return_value = mock_repo

        r = await client.post(
            "/v1/faces/enroll",
            data={"user_id": "user_1"},
            files={"image": ("face.png", _fake_image(), "image/png")},
        )

    assert r.status_code == 201
    body = r.json()
    assert body["user_id"] == "user_1"
    assert "embedding_id" in body


@pytest.mark.asyncio
async def test_enroll_no_face_detected(client: AsyncClient):
    with patch("app.api.routes.enroll.get_face_embedding_service") as mock_svc_factory:
        mock_svc = AsyncMock()
        mock_svc.get_embedding = AsyncMock(side_effect=ValueError("Nenhum rosto detectado na imagem."))
        mock_svc_factory.return_value = mock_svc

        r = await client.post(
            "/v1/faces/enroll",
            data={"user_id": "user_2"},
            files={"image": ("face.png", _fake_image(), "image/png")},
        )

    assert r.status_code == 422
    assert "rosto" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_verify_match(client: AsyncClient):
    emb = _fake_embedding()
    # Mesmo embedding = distância ~0
    distance = 0.05

    with (
        patch("app.api.routes.verify.get_face_embedding_service") as mock_svc_factory,
        patch("app.api.routes.verify.get_liveness_detector") as mock_live_factory,
        patch("app.api.routes.verify.FaceRepository") as MockRepo,
    ):
        mock_svc = AsyncMock()
        mock_svc.get_embedding = AsyncMock(return_value=emb)
        mock_svc_factory.return_value = mock_svc

        from app.services.liveness_service import LivenessResult
        mock_live = AsyncMock()
        mock_live.check = AsyncMock(return_value=LivenessResult(True, 99.0, "aws"))
        mock_live_factory.return_value = mock_live

        mock_repo = AsyncMock()
        mock_repo.exists = AsyncMock(return_value=True)
        mock_repo.find_closest = AsyncMock(return_value=(MagicMock(), distance))
        MockRepo.return_value = mock_repo

        r = await client.post(
            "/v1/faces/verify",
            data={"user_id": "user_1"},
            files={"image": ("face.png", _fake_image(), "image/png")},
        )

    assert r.status_code == 200
    body = r.json()
    assert body["match"] is True
    assert body["liveness_passed"] is True


@pytest.mark.asyncio
async def test_verify_liveness_fail(client: AsyncClient):
    emb = _fake_embedding()

    with (
        patch("app.api.routes.verify.get_face_embedding_service") as mock_svc_factory,
        patch("app.api.routes.verify.get_liveness_detector") as mock_live_factory,
        patch("app.api.routes.verify.FaceRepository") as MockRepo,
    ):
        mock_svc = AsyncMock()
        mock_svc.get_embedding = AsyncMock(return_value=emb)
        mock_svc_factory.return_value = mock_svc

        from app.services.liveness_service import LivenessResult
        mock_live = AsyncMock()
        mock_live.check = AsyncMock(return_value=LivenessResult(False, 20.0, "aws"))
        mock_live_factory.return_value = mock_live

        mock_repo = AsyncMock()
        mock_repo.exists = AsyncMock(return_value=True)
        MockRepo.return_value = mock_repo

        r = await client.post(
            "/v1/faces/verify",
            data={"user_id": "user_1"},
            files={"image": ("face.png", _fake_image(), "image/png")},
        )

    assert r.status_code == 200
    body = r.json()
    assert body["match"] is False
    assert body["liveness_passed"] is False


@pytest.mark.asyncio
async def test_verify_user_not_enrolled(client: AsyncClient):
    with patch("app.api.routes.verify.FaceRepository") as MockRepo:
        mock_repo = AsyncMock()
        mock_repo.exists = AsyncMock(return_value=False)
        MockRepo.return_value = mock_repo

        r = await client.post(
            "/v1/faces/verify",
            data={"user_id": "ghost_user"},
            files={"image": ("face.png", _fake_image(), "image/png")},
        )

    assert r.status_code == 404
