# NEODEMO3: Professional ML Project Structure

This project demonstrates a professional Python repository layout with robust dependency management, optimized specifically for Apple Silicon (arm64) and Python 3.12.

## Project Structure
- `src/`: Core logic and source code.
  - `loader.py`: Data ingestion.
  - `pipeline.py`: Model execution pipeline.
  - `benchmark.py`: Performance metrics.
  - `e2e_test.py`: End-to-end environment validation.
- `requirements.txt`: Pinned dependencies for reproducibility.
- `venv/`: Virtual environment (user-created).

## Setup Instructions

1. **Create and Activate Virtual Environment:**
   ```bash
   /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Verify Installation:**
   Run the end-to-end verification script to ensure all operators (like `torchvision::nms`) are functional:
   ```bash
   python3 -m src.e2e_test
   ```

## Usage
The project follows a standard `src-layout`. All scripts should be run as modules from the project root.
```bash
python3 -m src.pipeline
```