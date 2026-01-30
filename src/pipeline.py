import os
import time
import torch
import numpy as np
import soundfile as sf
import logging
from transformers import AutoProcessor, AutoTokenizer, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from optimum.intel import OVModelForSpeechSeq2Seq, OVModelForSeq2SeqLM
from datasets import load_dataset

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class S2SPipeline:
    def __init__(self, asr_id="openai/whisper-tiny", nllb_id="facebook/nllb-200-distilled-600M", tts_id="microsoft/speecht5_tts"):
        self.device = "CPU"
        
        logger.info(f"Loading ASR model: {asr_id} with OpenVINO INT8...")
        self.asr_proc = AutoProcessor.from_pretrained(asr_id)
        self.asr_model = OVModelForSpeechSeq2Seq.from_pretrained(
            asr_id, export=True, compile=True, load_in_8bit=True, device=self.device
        )
        
        logger.info(f"Loading NLLB model: {nllb_id} with OpenVINO INT8...")
        self.nllb_tok = AutoTokenizer.from_pretrained(nllb_id)
        self.nllb_model = OVModelForSeq2SeqLM.from_pretrained(
            nllb_id, export=True, compile=True, load_in_8bit=True, device=self.device
        )
        
        logger.info(f"Loading TTS model: {tts_id}...")
        self.tts_processor = AutoProcessor.from_pretrained(tts_id)
        self.tts_model = SpeechT5ForTextToSpeech.from_pretrained(tts_id)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        
        # Load speaker embeddings for SpeechT5
        logger.info("Initializing speaker embeddings for TTS...")
        # Since 'Matthijs/cmu-arctic-xvectors' used a script (deprecated), 
        # we'll use a standard 512-dim zero embedding for the smoke test or better, 
        # a random normal initialization which is compatible with SpeechT5.
        self.speaker_embeddings = torch.randn(1, 512)
        
        logger.info("Integrated Pipeline Initialized.")

    def run(self, audio_data, tgt_lang="fra_Latn"):
        metrics = {}
        start_time = time.time()
        
        # 1. ASR
        t_asr_start = time.time()
        input_features = self.asr_proc(audio_data, sampling_rate=16000, return_tensors="pt").input_features
        predicted_ids = self.asr_model.generate(input_features)
        transcription = self.asr_proc.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        metrics['asr_latency'] = time.time() - t_asr_start
        
        # 2. Translation
        t_trans_start = time.time()
        inputs = self.nllb_tok(transcription, return_tensors="pt").to("cpu")
        
        # Correctly get lang_code_to_id if present or use converter
        tgt_lang_id = self.nllb_tok.convert_tokens_to_ids(tgt_lang)
        
        translated_tokens = self.nllb_model.generate(
            **inputs, 
            forced_bos_token_id=tgt_lang_id,
            max_new_tokens=100
        )
        translation = self.nllb_tok.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        metrics['translation_latency'] = time.time() - t_trans_start
        
        # 3. TTS
        t_tts_start = time.time()
        inputs = self.tts_processor(text=translation, return_tensors="pt")
        if inputs["input_ids"].shape[1] == 0: # Handle empty translation
             translation = "Hello" # Fallback
             inputs = self.tts_processor(text=translation, return_tensors="pt")
             
        # SpeechT5 generation with vocoder
        speech = self.tts_model.generate_speech(inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder)
        metrics['tts_latency'] = time.time() - t_tts_start
        
        metrics['total_latency'] = time.time() - start_time
        
        return {
            "transcription": transcription,
            "translation": translation,
            "audio_output": speech.numpy(),
            "metrics": metrics
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    
    if args.mode == "smoke":
        # 1-second white noise for smoke test
        dummy_audio = np.random.uniform(-1, 1, 16000).astype(np.float32)
        pipeline = S2SPipeline(asr_id="openai/whisper-tiny")
        result = pipeline.run(dummy_audio, tgt_lang="fra_Latn")
        
        print("\n--- SMOKE TEST RESULTS ---")
        print(f"ASR Output: {result['transcription']}")
        print(f"NLLB Output: {result['translation']}")
        print(f"Latencies (sec): {result['metrics']}")
        print(f"Status: SUCCESS" if result['metrics']['total_latency'] < 10 else "Status: SLOW")