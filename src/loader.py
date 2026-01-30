import sys
import os

try:
    import openvino
    from optimum.intel import OVModelForSpeechSeq2Seq, OVModelForSeq2SeqLM
    print(f"SUCCESS: OpenVINO version {openvino.__version__} and optimum-intel are ready.")
except ImportError as e:
    print(f"FAILURE: {e}")
    sys.exit(1)

def load_optimized_models(asr_id="openai/whisper-tiny", nllb_id="facebook/nllb-200-distilled-600M"):
    """
    Check if we can initialize the optimized loaders.
    """
    print(f"Ready to export/load {asr_id} and {nllb_id} with OpenVINO INT8.")

if __name__ == "__main__":
    load_optimized_models()