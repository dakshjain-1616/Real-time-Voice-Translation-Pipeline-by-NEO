import os
import torch
import whisper
import time
import asyncio
import edge_tts
from transformers import MarianMTModel, MarianTokenizer
import argparse

class VoiceTranslationPipeline:
    def __init__(self, use_gpu=False):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"Initializing NEO Pipeline on {self.device}")
        
        # 1. STT: Whisper Tiny for <2s latency
        self.stt_model = whisper.load_model("tiny", device=self.device)
        
        # 2. MT: Cached models and tokenizers
        self.mt_models = {}
        self.mt_tokenizers = {}
        
        # 3. TTS: Edge-TTS (handled via async calls)
        self.supported_langs = ["en", "es", "fr", "de"]

    def _get_mt_model(self, src, tgt):
        key = f"{src}-{tgt}"
        if key not in self.mt_models:
            model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
            print(f"Loading MT model: {model_name}")
            self.mt_tokenizers[key] = MarianTokenizer.from_pretrained(model_name)
            self.mt_models[key] = MarianMTModel.from_pretrained(model_name).to(self.device)
            self.mt_models[key].eval()
        return self.mt_models[key], self.mt_tokenizers[key]

    def translate_text(self, text, src_lang, tgt_lang):
        if src_lang == tgt_lang or not text.strip():
            return text
        model, tokenizer = self._get_mt_model(src_lang, tgt_lang)
        tokens = tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            translated = model.generate(**tokens, max_new_tokens=64, num_beams=1, do_sample=False)
        return tokenizer.decode(translated[0], skip_special_tokens=True)

    async def text_to_speech(self, text, tgt_lang, output_path):
        voices = {
            "en": "en-US-GuyNeural",
            "es": "es-ES-AlvaroNeural",
            "fr": "fr-FR-HenriNeural",
            "de": "de-DE-ConradNeural"
        }
        voice = voices.get(tgt_lang, "en-US-GuyNeural")
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)

    def load_audio_whisper(self, audio_path):
        import librosa
        import numpy as np
        # Load audio and resample to 16kHz as required by Whisper
        audio, _ = librosa.load(audio_path, sr=16000)
        return audio.astype(np.float32)

    async def process_chunk(self, audio_path, src_lang, tgt_lang, out_audio_path):
        metrics = {}
        start_time = time.time()
        
        # STT Stage
        stt_start = time.time()
        audio_data = self.load_audio_whisper(audio_path)
        result = self.stt_model.transcribe(audio_data, fp16=(self.device=="cuda"))
        original_text = result['text'].strip()
        metrics['stt_time'] = time.time() - stt_start
        
        # Translation Stage
        mt_start = time.time()
        translated_text = self.translate_text(original_text, src_lang, tgt_lang)
        metrics['mt_time'] = time.time() - mt_start
        
        # TTS Stage
        tts_start = time.time()
        await self.text_to_speech(translated_text, tgt_lang, out_audio_path)
        metrics['tts_time'] = time.time() - tts_start
        
        metrics['total_latency'] = time.time() - start_time
        return original_text, translated_text, metrics

async def smoke_test():
    pipeline = VoiceTranslationPipeline()
    audio_input = "/Users/dakshjain/Desktop/GitHubDemos/NEODEMO3/data/en_clean.wav"
    output_dir = "/Users/dakshjain/Desktop/GitHubDemos/NEODEMO3/outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_audio = os.path.join(output_dir, "smoke_test_output.wav")
    
    if os.path.exists(audio_input):
        print("Warming up models...")
        # First run to load models
        await pipeline.process_chunk(audio_input, "en", "es", output_audio)
        
        print("Starting Smoke Test (Timed)...")
        orig, trans, metrics = await pipeline.process_chunk(audio_input, "en", "es", output_audio)
        print(f"Original (EN): {orig}")
        print(f"Translated (ES): {trans}")
        print(f"Metrics: {metrics}")
        if metrics['total_latency'] < 2.0:
            print("LATENCY CRITERIA MET (< 2.0s)")
        else:
            print(f"LATENCY CRITERIA FAILED: {metrics['total_latency']:.2f}s")
    else:
        print("Input file missing for smoke test.")

async def main():
    parser = argparse.ArgumentParser(description="NEO Real-time Voice Translation Pipeline")
    parser.add_argument("--test", action="store_true", help="Run the built-in smoke test")
    parser.add_argument("--input_file", type=str, help="Path to the input WAV audio file")
    parser.add_argument("--source_lang", type=str, default="en", help="Source language code (e.g., en, es, fr, de)")
    parser.add_argument("--target_lang", type=str, default="es", help="Target language code (e.g., en, es, fr, de)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save the translated audio")
    
    args = parser.parse_args()

    if args.test:
        await smoke_test()
    elif args.input_file:
        if not os.path.exists(args.input_file):
            print(f"Error: Input file not found at {args.input_file}")
            return

        os.makedirs(args.output_dir, exist_ok=True)
        filename = os.path.basename(args.input_file)
        output_filename = f"translated_{args.source_lang}_to_{args.target_lang}_{filename}"
        output_path = os.path.join(args.output_dir, output_filename)

        pipeline = VoiceTranslationPipeline()
        print(f"Processing: {args.input_file} ({args.source_lang} -> {args.target_lang})")
        
        orig, trans, metrics = await pipeline.process_chunk(
            args.input_file, 
            args.source_lang, 
            args.target_lang, 
            output_path
        )

        print("\n--- Translation Result ---")
        print(f"Original Text: {orig}")
        print(f"Translated Text: {trans}")
        print(f"Output Saved To: {output_path}")
        print(f"Total Latency: {metrics['total_latency']:.2f}s")
        print("--------------------------")
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())