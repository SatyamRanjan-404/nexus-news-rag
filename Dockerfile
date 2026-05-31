FROM python:3.10-slim

WORKDIR /app

# Copy requirements first (Docker layer caching — only re-installs if requirements change)
COPY requirement.txt .

# Install CPU-only torch first, then everything else
# The --extra-index-url tells pip where to find the +cpu builds
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirement.txt

# Copy the rest of your app
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]