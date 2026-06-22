FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
  apt-get install -y --no-install-recommends curl git && \
  rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"

# Copy project files
COPY . /app/

# Install Python dependencies (production only — skip dev group)
RUN uv sync --frozen --no-dev

# Create logs directory
RUN mkdir -p /app/logs

# Railway sets PORT environment variable
ENV PORT=8000

# Expose port (Railway will override with its own PORT)
EXPOSE 8000

# Start command (uses server.py which reads PORT from env or defaults to 8000)
CMD ["uv", "run", "python", "-m", "src.api.server"]
