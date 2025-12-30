# Dockerfile for secure code execution sandbox
# Used for dynamic verification of generated vulnerable code

FROM python:3.11-slim

# Security: Run as non-root user
RUN useradd -m -s /bin/bash sandbox
WORKDIR /home/sandbox

# Install minimal dependencies
RUN pip install --no-cache-dir \
    flask==3.0.0 \
    requests==2.31.0 \
    pyyaml==6.0.1 \
    lxml==5.1.0 \
    defusedxml==0.7.1

# Install semgrep for static analysis
RUN pip install --no-cache-dir semgrep==1.52.0

# Copy test harness
COPY --chown=sandbox:sandbox test_harness.py /home/sandbox/
COPY --chown=sandbox:sandbox vulnerability_rules.yaml /home/sandbox/

# Security restrictions
USER sandbox

# Network isolation (set at runtime with --network none)
# Resource limits (set at runtime with --memory, --cpus)

# Default command: run test harness
CMD ["python", "test_harness.py"]
