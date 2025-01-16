import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.integrate import cumulative_trapezoid

# Load the audio file
file_path = 'pm_demodulated.wav'
sample_rate, audio_data = wav.read(file_path)

# Normalize the audio data for visualization
audio_data = audio_data / np.max(np.abs(audio_data))

# Compute the derivative of the audio signal
derivative = np.diff(audio_data) * sample_rate

# Compute the integral of the audio signal
integral = cumulative_trapezoid(audio_data, dx=1/sample_rate, initial=0)

# Plot the original waveform
plt.figure(figsize=(12, 4))
plt.plot(audio_data, color='blue', alpha=0.7)
plt.title('Waveform of PM Demodulated Audio')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

# Plot the derivative of the waveform
plt.figure(figsize=(12, 4))
plt.plot(derivative, color='red', alpha=0.7)
plt.title('Derivative of PM Demodulated Audio')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

# Plot the integral of the waveform
plt.figure(figsize=(12, 4))
plt.plot(integral, color='green', alpha=0.7)
plt.title('Integral of PM Demodulated Audio')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

# Compute the spectrogram
frequencies, times, Sxx = spectrogram(audio_data, fs=sample_rate)

# Plot the spectrogram
plt.figure(figsize=(12, 6))
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.title('Spectrogram of PM Demodulated Audio')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.ylim(0, sample_rate // 2)
plt.show()
