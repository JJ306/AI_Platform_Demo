FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
ENV PYTHONUNBUFFERED=1 UV_SYSTEM_PYTHON=1

WORKDIR /app
COPY pyproject.toml /app/
COPY miniplat /app/miniplat
RUN uv pip install --system /app

RUN useradd -u 10001 -m appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "miniplat.main:app", "--host", "0.0.0.0", "--port", "8000"] 

