FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System build deps (required for PyRadiomics C extensions on arm64)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libgomp1 \
    dcm2niix \
  && rm -rf /var/lib/apt/lists/*

COPY deployment/requirements_deploy.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# PyRadiomics' setup.py imports numpy during metadata generation; ensure it is installed first.
RUN python -m pip install --no-cache-dir "numpy<2"

# Install everything except PyRadiomics first, then build PyRadiomics without build isolation
# (it imports numpy during wheel requirement discovery).
RUN grep -v -E '^pyradiomics' /tmp/requirements.txt > /tmp/requirements_no_pyradiomics.txt \
  && python -m pip install --no-cache-dir -r /tmp/requirements_no_pyradiomics.txt \
  && python -m pip install --no-cache-dir --no-build-isolation "pyradiomics==3.0.1"

COPY . /app

CMD ["python", "deployment/server.py"]
