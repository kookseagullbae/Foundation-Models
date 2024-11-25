import librosa
import matplotlib.pyplot as plt
import numpy as np

# Path to the audio file
path = "/Users/kookie/Desktop/audio data/output_repeated_splits_test_audio"

# Load the audio file
y, sr = librosa.load(path, sr=None)

# Trim the silent parts from the audio (reduces unnecessary zero padding)
y, _ = librosa.effects.trim(y)

# Normalize audio length to 2 seconds, note that sr is the sampling rate
VOICE_LEN_SECONDS = 2
target_len = VOICE_LEN_SECONDS * sr
nframes = len(y)

# If the audio is shorter than 2 seconds, pad with zeros
if nframes < target_len:
    res = target_len - nframes
    res_data = np.zeros(res, dtype=np.float32)
    y = np.concatenate((y, res_data))
# If the audio is longer than 2 seconds, truncate it
else:
    y = y[:target_len]

# Extract MFCC features
N_FFT = 512  # FFT window size
HOP_LENGTH = int(N_FFT / 4)  # Hop length between frames
mfcc_data = librosa.feature.mfcc(
    y=y, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=HOP_LENGTH
)

# Plot the MFCC feature matrix, transpose to make the time axis horizontal
plt.matshow(mfcc_data.T, aspect="auto", origin="lower")
plt.title(f"MFCC of {os.path.basename(path)}")
plt.xlabel("Time Frames")
plt.ylabel("MFCC Coefficients")
plt.colorbar()
plt.show()
