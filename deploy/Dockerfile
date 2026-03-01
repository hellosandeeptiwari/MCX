# ============================================================
#  Titan Trading Bot — Dockerfile
#  Lightweight Python image for trading bot + dashboard
# ============================================================
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    sqlite3 curl && \
    rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install Python deps first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy application code
COPY agentic_trader/ /app/agentic_trader/

# Create log + data directories
RUN mkdir -p /app/agentic_trader/logs /app/agentic_trader/data

# Non-root user
RUN useradd -m titan && chown -R titan:titan /app
USER titan

# Default: run the bot
WORKDIR /app/agentic_trader
CMD ["python", "autonomous_trader.py"]
