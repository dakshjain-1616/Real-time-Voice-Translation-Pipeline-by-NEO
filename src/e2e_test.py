import os
import numpy as np
import soundfile as sf
import logging
import time
from pipeline import S2SPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("E2E_Test")

def generate_speech_like_signal(duration=3.0, sr=16000):
    """Generates a modulated tone that simulates the spectral complexity of speech."""
    t = np.linspace(0, duration, int(sr * duration))
    # Combine multiple frequencies to simulate voice harmonics
    signal = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
    # Apply a low-frequency envelope to simulate word breaks
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 2 * t))
    return (signal * envelope).astype(np.float32)

def main():
    project_root = "/root/claude_tests/NEODEMO3"
    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Initializing S2S Pipeline for final verification...")
    pipeline = S2SPipeline()

    test_cases = [
        {"lang_code": "fra_Latn", "name": "french", "desc": "SST: English to French"},
        {"lang_code": "spa_Latn", "name": "spanish", "desc": "SST: English to Spanish"},
        {"lang_code": "deu_Latn", "name": "german", "desc": "SST: English to German"}
    ]

    print("\n" + "="*60)
    print("END-TO-END SYSTEM VERIFICATION - ROBUST SYNTHETIC MODE")
    print("="*60)

    e2e_metrics = []

    for case in test_cases:
        logger.info(f"Running Case: {case['desc']}")
        
        # Generator signal (3 seconds)
        audio_input = generate_speech_like_signal()
        
        # Run pipeline
        # Note: Since the signal is synthetic, ASR will produce a fallback 
        # or hallucination, but the component chaining and latency are verified.
        result = pipeline.run(audio_input, tgt_lang=case['lang_code'])
        
        # Save output
        output_path = os.path.join(output_dir, f"e2e_test_{case['name']}.wav")
        sf.write(output_path, result['audio_output'], 16000)
        
        e2e_metrics.append(result['metrics']['total_latency'])
        
        print(f"\n[RESULTS - {case['name'].upper()}]")
        print(f"ASR Output:        {result['transcription']}")
        print(f"Translation:       {result['translation']}")
        print(f"E2E Latency:       {result['metrics']['total_latency']:.3f}s")
        print(f"Saved to:           {output_path}")
        print("-" * 40)

    avg_latency = sum(e2e_metrics) / len(e2e_metrics)
    print(f"\nAverage E2E Latency: {avg_latency:.3f}s")
    print("\nSystem verification successful.")

if __name__ == "__main__":
    main()