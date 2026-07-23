# Minimal environment for A2A evaluation verifier
FROM python:3.12-slim

WORKDIR /workspace

# Install dependencies for LLM judge verifier
RUN pip install --no-cache-dir --root-user-action=ignore \
    litellm>=1.80.0 \
    pyyaml \
    requests

# Keep container running for Harbor
CMD ["sleep", "infinity"]
