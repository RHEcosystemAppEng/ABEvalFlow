# Minimal environment for A2A agent evaluation
# The actual agent runs as a sidecar container, this environment
# just provides the workspace for Harbor's verifier to run tests
FROM python:3.12-slim

WORKDIR /workspace

# Install minimal dependencies for verifier
RUN pip install --no-cache-dir pyyaml requests

# Copy instruction and tests
COPY instruction.md /workspace/
COPY tests/ /workspace/tests/

CMD ["sleep", "infinity"]
