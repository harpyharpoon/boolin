from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
import numpy as np

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Load the audio file
sample_rate, audio_data = read("extracted_audio.wav")

# Apply bandpass filter (4kHz to 6kHz)
filtered_audio = bandpass_filter(audio_data, lowcut=4000, highcut=6000, fs=sample_rate)

# Save filtered audio for listening
write("filtered_audio.wav", sample_rate, filtered_audio.astype(np.int16))

# Plot filtered waveform
plt.plot(filtered_audio)
plt.title("Filtered Audio Waveform (4kHz - 6kHz)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()
