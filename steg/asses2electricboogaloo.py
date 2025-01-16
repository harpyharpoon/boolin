import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Function to read and normalize audio data
def read_audio(file_path):
    rate, data = wavfile.read(file_path)
    if len(data.shape) > 1:
        data = data[:, 0]  # Use one channel if stereo
    data = data / np.max(np.abs(data))  # Normalize
    return rate, data

# Function to detect peaks in the audio data
def detect_peaks(data, height=0.5, distance=1000):
    peaks, _ = find_peaks(data, height=height, distance=distance)
    return peaks

# Function to analyze peaks for encoded data
def analyze_peaks(peaks, data):
    peak_values = data[peaks]
    print(f"Number of peaks detected: {len(peaks)}")
    print(f"Peak values: {peak_values}")

    # Example analysis: Check if peak values are above a certain threshold
    threshold = 0.7
    encoded_data = ''.join('1' if value > threshold else '0' for value in peak_values)
    print(f"Encoded data: {encoded_data}")

# Main function to read audio file, detect peaks, and analyze them
def main():
    file_path = '/home/harpoon/Documents/untitled.wav'  # Replace with your WAV file path
    rate, data = read_audio(file_path)
    
    peaks = detect_peaks(data)
    analyze_peaks(peaks, data)
    
    # Plot the waveform and detected peaks
    plt.figure(figsize=(16, 6))
    time = np.linspace(0, len(data) / rate, num=len(data))
    plt.plot(time, data, label="Waveform")
    plt.plot(time[peaks], data[peaks], 'x', label="Peaks")
    plt.title("Waveform with Detected Peaks")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()