# The mock toolchain images of the suite's own tests (see mock_toolchain.py). Build context: the repository root.
#   docker build --target clean -t oaax-mock-toolchain:1.0 -f mock-toolchain/Dockerfile .
# BASE only has to provide python3 (the manifest code needs nothing else; the ONNX parser is installed if missing).
ARG BASE=python:3.11-slim
FROM ${BASE} AS base
USER root
RUN python -c "import onnx" 2>/dev/null || pip install --no-cache-dir onnx
COPY python/oaax_conformance/manifest.py python/oaax_conformance/oaax-manifest.schema.json /app/oaax_conformance/
COPY mock-toolchain/mock_toolchain.py /app/mock_toolchain.py
RUN touch /app/oaax_conformance/__init__.py
WORKDIR /app

FROM base AS clean
USER 1000
ENTRYPOINT ["python", "/app/mock_toolchain.py"]

FROM base AS root
USER root
ENTRYPOINT ["python", "/app/mock_toolchain.py"]

FROM base AS noentry
USER 1000
ENTRYPOINT []
