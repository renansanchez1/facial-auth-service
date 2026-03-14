# 🔐 Facial Auth Service

Microserviço de reconhecimento facial para autenticação segura.  
Construído com **FastAPI + InsightFace (ArcFace) + pgvector**.

---

## Arquitetura

```
[Serviço de Login]
       │  POST /v1/faces/verify  (verifica na autenticação)
       │  POST /v1/faces/enroll  (cadastra no onboarding)
       ▼
[Facial Auth Service]  ←→  [PostgreSQL + pgvector]
       │
       └─→ [AWS Rekognition / Silent-Face]  (liveness check)
```

---

## Endpoints

### `POST /v1/faces/enroll`
Cadastra o rosto de um usuário.

**Form-data:**
| Campo    | Tipo   | Descrição                        |
|----------|--------|----------------------------------|
| `user_id`| string | ID do usuário no seu sistema     |
| `image`  | file   | Foto do rosto (JPEG/PNG/WebP)    |

**Response `201`:**
```json
{
  "user_id": "user_123",
  "embedding_id": "a1b2c3d4-...",
  "message": "Face cadastrada com sucesso."
}
```

---

### `POST /v1/faces/verify`
Verifica se o rosto corresponde ao usuário informado.

**Form-data:** igual ao enroll.

**Response `200`:**
```json
{
  "user_id": "user_123",
  "match": true,
  "confidence": 0.9742,
  "liveness_passed": true,
  "message": "Identidade verificada com sucesso."
}
```

> No seu serviço de login, use `match && liveness_passed` para liberar o acesso.

---

### `DELETE /v1/faces/enroll/{user_id}`
Remove o embedding do usuário (ex: ao excluir conta).

---

### `GET /health` e `GET /health/db`
Health checks para load balancer / Cloud Run.

---

## Rodando localmente

```bash
# 1. Clone e configure
cp .env.example .env

# 2. Suba banco + API
docker compose up --build

# 3. Acesse a documentação interativa
open http://localhost:8000/docs
```

---

## Variáveis de ambiente

| Variável                   | Padrão          | Descrição                                  |
|----------------------------|-----------------|--------------------------------------------|
| `DATABASE_URL`             | —               | PostgreSQL com asyncpg                     |
| `INSIGHTFACE_MODEL`        | `buffalo_l`     | Modelo InsightFace (buffalo_l = ArcFace R100) |
| `INSIGHTFACE_CTX_ID`       | `-1`            | -1 = CPU; 0 = GPU                          |
| `SIMILARITY_THRESHOLD`     | `0.40`          | Distância cosine máxima para match         |
| `LIVENESS_PROVIDER`        | `none`          | `aws` \| `silent_face` \| `none`           |
| `AWS_REGION`               | `us-east-1`     | Região do AWS Rekognition                  |
| `AWS_ACCESS_KEY_ID`        | —               | Credencial AWS                             |
| `AWS_SECRET_ACCESS_KEY`    | —               | Credencial AWS                             |
| `API_KEYS`                 | `[]`            | Lista JSON de API Keys válidas             |
| `RATE_LIMIT_REQUESTS`      | `10`            | Requisições por janela de tempo            |
| `RATE_LIMIT_WINDOW_SECONDS`| `60`            | Janela do rate limit em segundos           |

---

## Ajustando o threshold

O `SIMILARITY_THRESHOLD` controla o equilíbrio entre segurança e usabilidade:

| Threshold | FAR (falsos aceites) | FRR (falsas rejeições) |
|-----------|----------------------|-------------------------|
| `0.30`    | Baixo (mais seguro)  | Alto (mais restritivo)  |
| `0.40`    | ✅ Balanceado         | ✅ Balanceado            |
| `0.50`    | Alto (mais permissivo)| Baixo                  |

Para autenticação bancária/financeira use `0.30`. Para apps de consumo use `0.40–0.45`.

---

## Testes

```bash
pip install -r requirements.txt
pytest tests/ -v
```

---

## Deploy no Cloud Run

```bash
# Substitua PROJECT_ID e configure os secrets no Secret Manager antes
gcloud builds submit --config=cloudbuild.yaml \
  --substitutions=_REGION=us-central1 \
  --project=SEU_PROJECT_ID
```

---

## Segurança e LGPD

- **Apenas embeddings são armazenados** — nenhuma imagem é persistida.
- Embeddings são irreversíveis: não é possível reconstruir a imagem original.

---

## Integrando com seu serviço de login

```python
import httpx

async def facial_login(user_id: str, image_bytes: bytes) -> bool:
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://seu-facial-service/v1/faces/verify",
            data={"user_id": user_id},
            files={"image": ("frame.jpg", image_bytes, "image/jpeg")},
            headers={"X-API-Key": "sua-chave-aqui"},
            timeout=10.0,
        )
        r.raise_for_status()
        result = r.json()
        return result["match"] and result["liveness_passed"]
```
