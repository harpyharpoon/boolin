import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile

# Load the audio file again
file_path = '/home/harpoon/Documents/untitled.wav'
rate, data = wavfile.read(file_path)

# Handle stereo by selecting one channel if necessary
if len(data.shape) > 1:
    data = data[:, 0]

# Normalize data for better visualization
data = data / np.max(np.abs(data))

# Plot the waveform
plt.figure(figsize=(16, 6))
time = np.linspace(0, len(data) / rate, num=len(data))
plt.plot(time, data, label="Waveform")
plt.title("Waveform of the Audio File")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()
plt.tight_layout()

# Save the plot
waveform_path = "/home/harpoon/Documents/audio_waveform.png"
plt.savefig(waveform_path)
plt.close()

waveform_path
