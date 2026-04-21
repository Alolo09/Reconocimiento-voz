import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.io import wavfile
import librosa
from sklearn.cluster import KMeans



# ------------------ PARÁMETROS ------------------ #
FRAME_LENGTH = 320
HOP_LENGTH = 128
LPC_ORDER = 12
K = 32

# ------------------ FUNCIONES ------------------ #

def preemphasis(signal, alpha=0.95):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

def framing(signal):
    num_frames = int(np.floor((len(signal) - FRAME_LENGTH) / HOP_LENGTH)) + 1
    if num_frames <= 0:
        return np.array([])
    frames = []
    for i in range(num_frames):
        start = i * HOP_LENGTH
        frames.append(signal[start:start + FRAME_LENGTH])
    return np.array(frames)

def apply_hamming(frames):
    if len(frames) == 0:
        return frames
    window = np.hamming(FRAME_LENGTH)
    return frames * window

def compute_energy(frames):
    if len(frames) == 0:
        return np.array([])
    energy = np.sum(frames**2, axis=1)
    return np.log(energy + 1e-10)

# ----------- VAD ROBUSTO ----------- #
def endpoint_detection(energy, signal_len):
    if len(energy) == 0:
        return None, None

    # encontrar frame con máxima energía
    peak = np.argmax(energy)

    # tomar ventana alrededor del pico
    window = 8  # frames (~64 ms antes y después)

    start = max(peak - window, 0)
    end = min(peak + window, len(energy) - 1)

    start = start * HOP_LENGTH
    end = end * HOP_LENGTH + FRAME_LENGTH

    return start, min(end, signal_len)

# ----------- LPC ----------- #
def compute_lpc(frames):
    lpcs = []

    for f in frames:
        # asegurar tipo correcto
        f = f.astype(np.float64)

        # normalizar frame
        max_val = np.max(np.abs(f))
        if max_val < 1e-6:
            continue

        f = f / max_val

        # agregar pequeño ruido (evita matrices singulares)
        f = f + 1e-6

        try:
            a = librosa.lpc(y=f, order=LPC_ORDER)

            # validar resultado
            if np.any(np.isnan(a)) or np.any(np.isinf(a)):
                continue

            lpcs.append(a[1:])

        except Exception:
            continue

    return np.array(lpcs)

# ----------- PIPELINE COMPLETO ----------- #
def extract_features(file):
    fs, signal = wavfile.read(file)
    signal = signal.astype(np.float32)

    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    print(f"\nProcesando: {file}")
    print(f"Longitud señal: {len(signal)}")

    # normalización
    max_val = np.max(np.abs(signal))
    print(f"Max valor: {max_val}")

    if max_val < 1e-5:
        print("❌ Señal casi cero")
        return None

    signal = signal / max_val

    # preénfasis
    signal_pre = preemphasis(signal)

    # SIN VAD — usamos toda la señal
    frames = framing(signal_pre)

    print(f"Frames generados: {len(frames)}")

    if len(frames) == 0:
        print("❌ No hay frames")
        return None

    frames = apply_hamming(frames)

    lpc = compute_lpc(frames)

    print(f"LPC frames: {len(lpc)}")

    if len(lpc) == 0:
        print("❌ LPC falló")
        return None

    return lpc



# ----------- DISTANCIA ----------- #
def distance_to_codebook(features, codebook):
    total = 0
    for vec in features:
        d = np.min(np.linalg.norm(codebook - vec, axis=1))
        total += d
    return total

# ------------------ MAIN ------------------ #

files = sorted(glob.glob("dataset/start/*.wav"))

train_files = files[:10]
test_files = files[10:]

print(f"Train: {len(train_files)} | Test: {len(test_files)}\n")

# ----------- ENTRENAMIENTO ----------- #
train_features = []

for f in train_files:
    feats = extract_features(f)

    if feats is None:
        print(f"{f} → ❌ VAD o extracción falló")
        continue

    if len(feats) == 0:
        print(f"{f} → ❌ sin features")
        continue

    print(f"{f} → ✔ OK ({len(feats)} frames)")
    train_features.append(feats)

# PROTECCIÓN
if len(train_features) == 0:
    raise ValueError("No se extrajeron features de ningún archivo. Revisa VAD.")

train_features = np.vstack(train_features)

print("\nEntrenando codebook...\n")

kmeans = KMeans(n_clusters=K, random_state=0)
kmeans.fit(train_features)

codebook = kmeans.cluster_centers_

# ----------- EVALUACIÓN ----------- #
print("Evaluación:\n")

for f in test_files:
    feats = extract_features(f)

    if feats is None:
        print(f"{f} → ❌ ERROR en VAD")
        continue

    dist = distance_to_codebook(feats, codebook)
    print(f"{f} → distancia: {dist:.2f}")

# visualización de un ejemplo

file = "dataset/start/start_11.wav"

fs, signal = wavfile.read(file)
signal = signal.astype(np.float32)

if signal.ndim > 1:
    signal = signal.mean(axis=1)

t = np.arange(len(signal)) / fs

plt.figure(figsize=(10,4))
plt.plot(t, signal)
plt.title("Señal de voz (ejemplo: 'start')")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid()
plt.show()