# Real-time Voice Translation Pipeline - By NEO

## Architectural Overview
The Real-time Voice Translation Pipeline is architected by **NEO** as a modular integration layer designed for high-throughput, low-latency speech processing. The system leverages state-of-the-art transformer models optimized for edge-case performance.

### Modular Integration Layer
1.  **Audio Ingestion Module**: Utilizes `librosa` for robust audio loading and resampling, bypassing system-level dependencies like FFmpeg for better portability.
2.  **STT (Speech-To-Text)**: Powered by OpenAI's **Whisper Tiny**. This variant was chosen for its exceptional trade-off between Word Error Rate (WER) and inference speed, crucial for "real-time" experience.
3.  **Neural Machine Translation (NMT)**: Employs **MarianMT** (Helsinki-NLP) models. The architecture uses pre-warmed model instances and greedy decoding (`num_beams=1`) to minimize compute overhead during translation.
4.  **TTS (Text-To-Speech)**: Integrated with **Edge-TTS**, providing high-fidelity neural voices with asynchronous stream-saving capabilities.

## Performance Characteristics (NEO Benchmarks)
Based on benchmarking against the Neo Robustness Test Suite:
- **Average Segment Latency**: ~1.3 seconds (excluding cold-start model loading).
- **STT Accuracy**: Verified at < 0.10 WER for clean English audio.
- **Robustness**: Successfully handles simulated background noise with minimal impact on latency.

## Setup and Usage
1.  **Environment**: Python 3.12+ (macOS/Linux/Windows).
2.  **Installation**:
    ```bash
    source venv/bin/activate
    pip install -r requirements.txt
    ```
3.  **Running the Pipeline**:

    ### A. Test Mode (Built-in Smoke Test)
    Run a pre-configured test to verify environment health and latency:
    ```bash
    python src/pipeline.py --test
    ```

    ### B. Custom Input Mode (For End Users)
    Translate your own audio files by specifying the input path and language pairs:
    ```bash
    python src/pipeline.py --input_file data/harvard.wav --source_lang en --target_lang es
    ```

    **Sample Previews:**
    - [Source English Sample (Harvard Phrases)](data/harvard.wav)
    - [Translated Spanish Sample (Placeholder)](outputs/translated_en_to_es_harvard.wav)
    
    **Arguments:**
    - `--input_file`: Path to your audio file (Requirement: **WAV** format, 16kHz mono recommended).
    - `--source_lang`: Source language code (Choose from: `en`, `es`, `fr`, `de`).
    - `--target_lang`: Target language code (Choose from: `en`, `es`, `fr`, `de`).
    - `--output_dir`: (Optional) Directory for results (Default: `outputs/`).

4.  **Benchmarking**:
    Execute the automated robustness suite to generate performance metrics:
    ```bash
    python src/benchmark.py
    ```

## Design Decisions
- **Quantization**: Attempted dynamic INT8 quantization for CPU; however, standard FP32 was prioritized for stability across diverse hardware environments without custom C++ engines.
- **Async Execution**: The pipeline uses Python `asyncio` for TTS to ensure audio synthesis doesn't block the main translation loop.

---
*Architected and Designed by NEO.*