import numpy as np
from scipy.io.wavfile import read, write
from scipy.signal import hilbert, resample, stft, spectrogram
from scipy.fft import fft, ifft
import pywt
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

# Function to demodulate FM
def demodulate_fm(audio_data, sample_rate):
    analytic_signal = hilbert(audio_data)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * sample_rate
    normalized_freq = instantaneous_frequency / np.max(np.abs(instantaneous_frequency))
    return normalized_freq

# Function to demodulate AM
def demodulate_am(audio_data):
    analytic_signal = hilbert(audio_data)
    envelope = np.abs(analytic_signal)
    return envelope

# Function to demodulate PM
def demodulate_pm(audio_data):
    analytic_signal = hilbert(audio_data)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    return instantaneous_phase

# Function to perform STFT
def perform_stft(audio_data, sample_rate):
    frequencies, times, Zxx = stft(audio_data, fs=sample_rate, nperseg=1024)
    return frequencies, times, Zxx

# Function to perform Wavelet Transform
def perform_wavelet_transform(audio_data, sample_rate):
    scales = np.arange(1, 128)
    coeffs, freqs = pywt.cwt(audio_data, scales, 'morl', sampling_period=1/sample_rate)
    return coeffs, freqs

# Function to perform Cepstrum Analysis
def perform_cepstrum_analysis(audio_data):
    spectrum = fft(audio_data)
    log_spectrum = np.log(np.abs(spectrum) + 1e-10)
    cepstrum = np.real(ifft(log_spectrum))
    return cepstrum

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
def compute_zero_crossing_rate(audio_data, frame_size, hop_size):
    zcr = []
    for start in range(0, len(audio_data) - frame_size, hop_size):
        frame = audio_data[start:start + frame_size]
        zero_crossings = np.where(np.diff(np.sign(frame)))[0]
        zcr.append(len(zero_crossings) / frame_size)
    return np.array(zcr)

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
    
    # Perform FM demodulation
    fm_demodulated = demodulate_fm(audio_data, sample_rate)
    fm_demodulated = fm_demodulated / np.max(np.abs(fm_demodulated))  # Normalize
    write("fm_demodulated.wav", sample_rate, (fm_demodulated * 32767).astype(np.int16))
    plt.plot(fm_demodulated[:sample_rate])  # Show the first second
    plt.title("FM Demodulated Signal")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.savefig("fm_demodulated.png")
    plt.close()
    
    # Perform AM demodulation
    am_demodulated = demodulate_am(audio_data)
    am_demodulated = am_demodulated / np.max(np.abs(am_demodulated))  # Normalize
    write("am_demodulated.wav", sample_rate, (am_demodulated * 32767).astype(np.int16))
    plt.plot(am_demodulated[:sample_rate])  # Show the first second
    plt.title("AM Demodulated Signal")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.savefig("am_demodulated.png")
    plt.close()
    
    # Perform PM demodulation
    pm_demodulated = demodulate_pm(audio_data)
    pm_demodulated = pm_demodulated / np.max(np.abs(pm_demodulated))  # Normalize
    write("pm_demodulated.wav", sample_rate, (pm_demodulated * 32767).astype(np.int16))
    plt.plot(pm_demodulated[:sample_rate])  # Show the first second
    plt.title("PM Demodulated Signal")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.savefig("pm_demodulated.png")
    plt.close()
    
    # Perform STFT
    frequencies, times, Zxx = perform_stft(audio_data, sample_rate)
    plt.pcolormesh(times, frequencies, np.abs(Zxx), shading='gouraud')
    plt.title("STFT Magnitude")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar(label='Magnitude')
    plt.savefig("stft_magnitude.png")
    plt.close()
    
    # Perform Wavelet Transform
    coeffs, freqs = perform_wavelet_transform(audio_data, sample_rate)
    plt.imshow(np.abs(coeffs), extent=[0, len(audio_data)/sample_rate, 1, 128], cmap='PRGn', aspect='auto',
               vmax=abs(coeffs).max(), vmin=-abs(coeffs).max())
    plt.title("Wavelet Transform (CWT)")
    plt.xlabel("Time [s]")
    plt.ylabel("Scale")
    plt.colorbar(label='Magnitude')
    plt.savefig("wavelet_transform.png")
    plt.close()
    
    # Perform Cepstrum Analysis
    cepstrum = perform_cepstrum_analysis(audio_data)
    plt.plot(cepstrum[:sample_rate])  # Show the first second
    plt.title("Cepstrum")
    plt.xlabel("Quefrency [samples]")
    plt.ylabel("Amplitude")
    plt.savefig("cepstrum.png")
    plt.close()
    
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
    frame_size = 1024
    hop_size = 512
    zcr = compute_zero_crossing_rate(audio_data, frame_size, hop_size)
    plt.plot(zcr)
    plt.title("Zero-Crossing Rate Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Zero-Crossing Rate")
    plt.savefig("zero_crossing_rate.png")
    plt.close()
    
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