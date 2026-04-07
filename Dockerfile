FROM python:3.12-slim

WORKDIR /app

# Install PyTorch CPU-only and PyG
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    torch-geometric \
    requests \
    numpy \
    pytest \
    pytest-mock

# Copy project
COPY pyproject.toml .
COPY src/ src/
COPY tests/ tests/
COPY data/ data/
COPY README.md .

# Install package
RUN pip install --no-cache-dir -e .

# Output goes here by default -- mount this to get files out
RUN mkdir -p /app/output
VOLUME /app/output

# Default: run tests
CMD ["pytest", "tests/", "-v"]
