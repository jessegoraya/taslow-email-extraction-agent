FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md requirements.txt ./
COPY src ./src

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["uvicorn", "taslow_email_extraction_agent.app:app", "--host", "0.0.0.0", "--port", "8080"]

