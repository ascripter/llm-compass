FROM python:3.13-slim

WORKDIR /app

# Copy project files
COPY pyproject.toml README.md ./
COPY src ./src

# Install dependencies from pyproject.toml
RUN pip install --no-cache-dir -e .

# Set PYTHONPATH
ENV PYTHONPATH=/app/src

CMD ["python", "-m", "llm_compass.app"]