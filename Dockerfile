FROM python:3.12-slim
WORKDIR /suite
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml ./
COPY app ./app
COPY scripts ./scripts
RUN pip install --no-cache-dir '.[ml]' --extra-index-url https://download.pytorch.org/whl/cpu
COPY public ./public
COPY reports ./reports
ENV ENABLE_TRANSFORMERS=1
RUN useradd --create-home suite && chown -R suite:suite /suite
USER suite
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
