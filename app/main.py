import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import enroll, verify, health
from app.core.config import settings
from app.core.database import init_db
from app.core.logger import logger
from app.core.rate_limiter import RateLimiter

rate_limiter = RateLimiter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Facial Auth Service...")
    await init_db()
    logger.info("Database initialized.")
    yield
    logger.info("Shutting down Facial Auth Service.")


app = FastAPI(
    title="Facial Auth Service",
    description="Microserviço de reconhecimento facial para autenticação segura.",
    version="1.0.0",
    docs_url="/docs" if settings.ENV != "production" else None,
    redoc_url="/redoc" if settings.ENV != "production" else None,
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Middlewares
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "GET", "DELETE"],
    allow_headers=["*"],
)

if settings.ENV == "production":
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.ALLOWED_HOSTS)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{elapsed:.2f}ms"
    return response


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    allowed, retry_after = await rate_limiter.is_allowed(client_ip)
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests.", "retry_after_seconds": retry_after},
            headers={"Retry-After": str(retry_after)},
        )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(health.router, tags=["Health"])
app.include_router(enroll.router, prefix="/v1/faces", tags=["Enrollment"])
app.include_router(verify.router, prefix="/v1/faces", tags=["Verification"])
