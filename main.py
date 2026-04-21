import os
import sounddevice as sd
from scipy.io.wavfile import write, read
import numpy as np
import time

# frecuencia de muestreo de 16 kHz, duración de 1.5 segundos 
# Un solo canal (mono) para la grabación de audio. 
# El audio se guarda en formato WAV con una profundidad de bits de 16 bits por muestra.

fs = 16000
duration = 2.0 
silence_threshold = 500  # Umbral para detectar silencio (ajustable según el entorno)

words = ["start","stop","left","right","forward","back","lift","drop","faster","slower"]
reps = 15
base_dir = "dataset"

def record_once(): 
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()  
    return audio.flatten()

def playback(audio):
    print("Reproduciendo grabación para revisión...")
    sd.play(audio, samplerate=fs)
    sd.wait()

def check_quality(audio):
    peak = np.max(np.abs(audio))
    if peak < silence_threshold:
        print("Advertencia: Grabación demasiado silenciosa. Intenta hablar más fuerte.")
        return False
    elif peak > 30000:
        print("Advertencia: Grabación demasiado fuerte. Intenta hablar más suavemente.")
        return False
    return True

def record_with_retry(word, rep, path):
    while True:
        input(f"Presiona ENTER para grabar '{word}' ({rep}/{reps})")
        audio = record_once()
        print("Grabación completa.")

        if not check_quality(audio):
             continue
        
        choice = input("[ENTER] Guardar, [R] Regrabar, [P] Reproducir: ").strip().lower()
        if choice == 'r':
            continue
        if choice == 'p':
            print("Reproduciendo...")
            playback(audio)
            choice2 = input("[ENTER] Guardar, [R] Regrabar: ").strip().lower()
            if choice2 == 'r':
                continue

        write(path, fs, audio)
        print(f"Guardado; {path} \n")
        break

# Crear directorio base si no existe
os.makedirs(base_dir, exist_ok=True)
saved, skipped = 0, 0

for word in words:
    word_dir = os.path.join(base_dir, word)
    os.makedirs(word_dir, exist_ok=True)

    print(f"\n{'─'*40}")
    print(f"  Palabra: '{word.upper()}'  ({reps} grabaciones)")
    print(f"{'─'*40}")

    for i in range(1, reps + 1):
        path = os.path.join(word_dir, f"{word}_{i:02d}.wav")

        # Si el archivo ya existe, preguntar si regrabar
        if os.path.exists(path):
            skip = input(f"  '{path}' ya existe. [ENTER] Saltar · [R] Regrabar: ").strip().lower()
            if skip != 'r':
                skipped += 1
                continue

        record_with_retry(word, i, path)
        
        saved += 1

print(f"\n{'='*40}")
print(f"  Dataset completo.")
print(f"  Archivos nuevos: {saved}  |  Saltados: {skipped}")
print(f"{'='*40}")