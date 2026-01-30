# Deployment Guide: NEODEMO3

This document provides instructions for deploying and running the NEODEMO3 S2S pipeline in different environments.

## Deployment Requirements
- **Hardware**: Minimum 4-core CPU, 8GB RAM recommended for OpenVINO inference.
- **Software**: Linux/macOS/Windows with Python 3.12+.

## Local Deployment
1. **Environment Setup**:
   ```bash
   python -m venv .vh
   source .vh/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. **Model Caching**:
   The first run of the pipeline will download models (~1.5GB) to the HuggingFace cache directory (usually `~/.cache/huggingface`).

## CI/CD and Linting
The project is configured for PEP 8 compliance. You can use `flake8` or `black` for linting:
```bash
pip install black
black src/
```

## Production Considerations
- **OpenVINO Optimization**: Models are exported to OpenVINO format on the first run. For production, consider pre-exporting and freezing the IR (.xml/.bin) files.
- **Scaling**: For high-throughput scenarios, deploy as a containerized microservice behind a task queue (e.g., Celery/Redis).