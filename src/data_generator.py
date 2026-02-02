import os
import json
import asyncio
import edge_tts
import numpy as np
import soundfile as sf
import librosa

async def generate_audio(text, lang, output_path, voice):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

def add_noise(audio_path, output_path, noise_level=0.005):
    y, sr = librosa.load(audio_path, sr=None)
    noise = np.random.randn(len(y))
    y_noisy = y + noise_level * noise
    sf.write(output_path, y_noisy, sr)

async def main():
    data_dir = "/Users/dakshjain/Desktop/GitHubDemos/NEODEMO3/data"
    os.makedirs(data_dir, exist_ok=True)
    
    test_cases = [
        {"id": "en_clean", "text": "Hello, how are you today?", "lang": "en", "voice": "en-US-GuyNeural", "category": "clean"},
        {"id": "es_clean", "text": "Hola, ¿cómo estás hoy?", "lang": "es", "voice": "es-ES-AlvaroNeural", "category": "clean"},
        {"id": "fr_clean", "text": "Bonjour, comment allez-vous aujourd'hui?", "lang": "fr", "voice": "fr-FR-HenriNeural", "category": "clean"},
        {"id": "de_clean", "text": "Hallo, wie geht es dir heute?", "lang": "de", "voice": "de-DE-ConradNeural", "category": "clean"},
    ]
    
    metadata = {}
    
    for case in test_cases:
        clean_path = os.path.join(data_dir, f"{case['id']}.wav")
        await generate_audio(case['text'], case['lang'], clean_path, case['voice'])
        
        metadata[case['id']] = {
            "file": f"{case['id']}.wav",
            "transcript": case['text'],
            "lang": case['lang'],
            "category": "clean",
            "translations": {
                "en": "Hello, how are you today?",
                "es": "Hola, ¿cómo estás hoy?",
                "fr": "Bonjour, comment allez-vous aujourd'hui?",
                "de": "Hallo, wie geht es dir heute?"
            }
        }
        
        # Simulated Background Noise
        noisy_id = case['id'].replace("clean", "noise")
        noisy_path = os.path.join(data_dir, f"{noisy_id}.wav")
        add_noise(clean_path, noisy_path)
        
        metadata[noisy_id] = {
            "file": f"{noisy_id}.wav",
            "transcript": case['text'],
            "lang": case['lang'],
            "category": "background-noise",
            "translations": metadata[case['id']]["translations"]
        }

    with open(os.path.join(data_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Generated {len(metadata)} test samples in {data_dir}")

if __name__ == "__main__":
    asyncio.run(main())