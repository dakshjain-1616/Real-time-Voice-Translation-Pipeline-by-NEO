import librosa
import numpy as np
import io
import soundfile as sf

def load_audio(audio_input, target_sr=16000):
    """
    Load audio from file path or buffer and resample to target_sr.
    """
    if isinstance(audio_input, str):
        audio, _ = librosa.load(audio_input, sr=target_sr)
    elif isinstance(audio_input, bytes):
        audio_fp = io.BytesIO(audio_input)
        audio, _ = librosa.load(audio_fp, sr=target_sr)
    elif isinstance(audio_input, np.ndarray):
        audio = audio_input
    else:
        raise ValueError("Unsupported audio input type")
    
    return audio

def save_audio(audio, path, sr=16000):
    """Save numpy array to wav file."""
    sf.write(path, audio, sr)

if __name__ == "__main__":
    print("Loader module ready.")