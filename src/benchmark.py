import os
import json
import time
import asyncio
import numpy as np
import sacrebleu
from jiwer import wer
from pipeline import VoiceTranslationPipeline

async def run_benchmark():
    data_dir = "/Users/dakshjain/Desktop/GitHubDemos/NEODEMO3/data"
    metadata_path = os.path.join(data_dir, "metadata.json")
    output_dir = "/Users/dakshjain/Desktop/GitHubDemos/NEODEMO3/outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    pipeline = VoiceTranslationPipeline()
    results = []
    
    print(f"Starting Benchmark on {len(metadata)} samples...")
    
    # Warmup
    warmup_file = os.path.join(data_dir, "en_clean.wav")
    await pipeline.process_chunk(warmup_file, "en", "es", os.path.join(output_dir, "warmup.wav"))

    for sample_id, info in metadata.items():
        audio_path = os.path.join(data_dir, info['file'])
        src_lang = info['lang']
        # For simplicity, benchmark translation to Spanish for EN, and to English for others
        tgt_lang = "es" if src_lang == "en" else "en"
        
        out_audio = os.path.join(output_dir, f"bench_{sample_id}.wav")
        
        orig_text, trans_text, metrics = await pipeline.process_chunk(audio_path, src_lang, tgt_lang, out_audio)
        
        # Calculate WER for STT
        ground_truth = info['transcript'].lower()
        pred_text = orig_text.lower()
        error_rate = wer(ground_truth, pred_text)
        
        # Calculate BLEU for Translation (if target translation exists in metadata)
        ref_translation = info['translations'].get(tgt_lang, "")
        bleu_score = 0
        if ref_translation:
            bleu_score = sacrebleu.sentence_bleu(trans_text, [ref_translation]).score
            
        results.append({
            "id": sample_id,
            "category": info['category'],
            "metrics": metrics,
            "wer": error_rate,
            "bleu": bleu_score
        })
        print(f"Finished {sample_id} | Latency: {metrics['total_latency']:.2f}s | WER: {error_rate:.2f}")

    # Aggregated Stats
    avg_latency = np.mean([r['metrics']['total_latency'] for r in results])
    avg_stt = np.mean([r['metrics']['stt_time'] for r in results])
    avg_mt = np.mean([r['metrics']['mt_time'] for r in results])
    avg_tts = np.mean([r['metrics']['tts_time'] for r in results])
    avg_wer = np.mean([r['wer'] for r in results])
    avg_bleu = np.mean([r['bleu'] for r in results])

    report = f"""Real-time Voice Translation Pipeline - Benchmark Report
======================================================
Total Samples: {len(results)}
Average End-to-End Latency: {avg_latency:.2f}s
Breakdown:
  - STT (Whisper): {avg_stt:.2f}s
  - MT (MarianMT): {avg_mt:.2f}s
  - TTS (Edge-TTS): {avg_tts:.2f}s

Performance Metrics:
  - Average WER (STT): {avg_wer:.4f}
  - Average BLEU (Translation): {avg_bleu:.2f}

Category Wise Latency (Avg):
  - Clean: {np.mean([r['metrics']['total_latency'] for r in results if r['category'] == 'clean']):.2f}s
  - Noisy: {np.mean([r['metrics']['total_latency'] for r in results if r['category'] == 'background-noise']):.2f}s
======================================================
"""
    with open("/Users/dakshjain/Desktop/GitHubDemos/NEODEMO3/benchmark_report.txt", "w") as f:
        f.write(report)
    print("Benchmark complete. Report saved to benchmark_report.txt")

if __name__ == "__main__":
    asyncio.run(run_benchmark())