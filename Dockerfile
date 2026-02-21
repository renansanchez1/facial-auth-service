FROM python:3.11-slim

# Dependências do sistema para InsightFace / OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instala dependências Python antes de copiar o código (cache de layers)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copia o código
COPY . .

# Usuário não-root
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser /app
USER appuser

# InsightFace baixa modelos na primeira execução para ~/.insightface
# Em produção, pré-baixe e inclua na imagem para startup mais rápido:
# COPY models/ /home/appuser/.insightface/models/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
