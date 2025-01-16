import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

def analyze_audio(file_path):
    # Step 1: Load the Audio File
    sample_rate, audio_data = wavfile.read(file_path)
    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Audio Length: {len(audio_data) / sample_rate:.2f} seconds")

    # Step 2: Generate a Spectrogram
    plt.figure(figsize=(10, 6))
    f, t, Sxx = spectrogram(audio_data, sample_rate, nperseg=1024, noverlap=512, scaling='spectrum')
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
    plt.title("Spectrogram")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.colorbar(label="Power [dB]")
    plt.show()

    # Step 3: Plot the Waveform
    plt.figure(figsize=(10, 4))
    time = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))
    plt.plot(time, audio_data, linewidth=0.5)
    plt.title("Waveform")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

    # Step 4: Perform Fourier Transform
    fft_data = np.fft.fft(audio_data)
    fft_freq = np.fft.fftfreq(len(fft_data), d=1/sample_rate)
    magnitude = np.abs(fft_data)

    # Plot Frequency Spectrum
    plt.figure(figsize=(10, 4))
    plt.plot(fft_freq[:len(fft_freq)//2], magnitude[:len(magnitude)//2], linewidth=0.5)
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.show()

# Run the analysis
audio_path = "extracted_audio.wav"  # Replace with the path to your audio file
analyze_audio(audio_path)
