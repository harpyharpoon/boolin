import numpy as np
from scipy.io.wavfile import read, write
from scipy.signal import hilbert, resample, stft, spectrogram
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt

# Use a non-interactive backend for matplotlib
plt.switch_backend('Agg')

# Function to read and normalize audio data
def read_audio(file_path):
    rate, data = read(file_path)
    if len(data.shape) > 1:
        data = data[:, 0]  # Use one channel if stereo
    data = data / np.max(np.abs(data))  # Normalize
    return rate, data

# Function to resample audio data
def resample_audio(data, original_rate, target_rate):
    num_samples = int(len(data) * float(target_rate) / original_rate)
    resampled_data = resample(data, num_samples)
    return resampled_data, target_rate

# Function to detect the envelope of the signal
def detect_envelope(audio_data):
    analytic_signal = hilbert(audio_data)
    envelope = np.abs(analytic_signal)
    return envelope

# Function to compute autocorrelation
def compute_autocorrelation(audio_data):
    autocorr = np.correlate(audio_data, audio_data, mode='full')
    return autocorr[autocorr.size // 2:]

# Function to compute zero-crossing rate
def compute_zero_crossing_rate(audio_data):
    zero_crossings = np.where(np.diff(np.sign(audio_data)))[0]
    return len(zero_crossings) / len(audio_data)

# Function to perform frequency analysis
def frequency_analysis(audio_data, sample_rate):
    spectrum = np.abs(fft(audio_data))
    freqs = np.fft.fftfreq(len(spectrum), 1/sample_rate)
    return freqs, spectrum

# Main function to read audio file and apply different transformations
def main():
    file_path = '/home/harpoon/Documents/steg/extracted_audio.wav'  # Replace with your WAV file path
    sample_rate, audio_data = read_audio(file_path)
    
    # Resample the audio to a lower sample rate for better peak detection
    target_rate = 8000  # Target sample rate (adjust as needed)
    audio_data, sample_rate = resample_audio(audio_data, sample_rate, target_rate)
    
    # Detect envelope
    envelope = detect_envelope(audio_data)
    write("envelope.wav", sample_rate, (envelope * 32767).astype(np.int16))
    plt.plot(envelope[:sample_rate])  # Show the first second
    plt.title("Envelope of the Signal")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.savefig("envelope.png")
    plt.close()
    
    # Compute autocorrelation
    autocorr = compute_autocorrelation(audio_data)
    plt.plot(autocorr[:sample_rate])  # Show the first second
    plt.title("Autocorrelation of the Signal")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.savefig("autocorrelation.png")
    plt.close()
    
    # Compute zero-crossing rate
    zcr = compute_zero_crossing_rate(audio_data)
    print(f"Zero-Crossing Rate: {zcr}")
    
    # Perform frequency analysis
    freqs, spectrum = frequency_analysis(audio_data, sample_rate)
    plt.plot(freqs[:len(freqs)//2], spectrum[:len(spectrum)//2])  # Plot only positive frequencies
    plt.title("Frequency Analysis")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.savefig("frequency_analysis.png")
    plt.close()
    
    # Compute spectrogram
    frequencies, times, Sxx = spectrogram(audio_data, fs=sample_rate)
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
    plt.title("Spectrogram")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar(label='Intensity [dB]')
    plt.savefig("spectrogram.png")
    plt.close()

if __name__ == "__main__":
    main()