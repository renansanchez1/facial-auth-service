from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enrollment
# ---------------------------------------------------------------------------

class EnrollRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=255, description="ID único do usuário no seu sistema.")


class EnrollResponse(BaseModel):
    user_id: str
    embedding_id: str
    message: str = "Face cadastrada com sucesso."


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

class VerifyRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=255)


class VerifyResponse(BaseModel):
    user_id: str
    match: bool
    confidence: float = Field(..., description="Score de similaridade [0–1]. Mais alto = mais similar.")
    liveness_passed: bool
    message: str


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

class DeleteResponse(BaseModel):
    user_id: str
    deleted_count: int
    message: str


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    detail: str
    code: Optional[str] = None
