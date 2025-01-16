import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, resample
from scipy.fft import fft, ifft

# Function to read and normalize audio data
def read_audio(file_path):
    rate, data = wavfile.read(file_path)
    if len(data.shape) > 1:
        data = data[:, 0]  # Use one channel if stereo
    data = data / np.max(np.abs(data))  # Normalize
    return rate, data

# Function to resample audio data
def resample_audio(data, original_rate, target_rate):
    num_samples = int(len(data) * float(target_rate) / original_rate)
    resampled_data = resample(data, num_samples)
    return resampled_data, target_rate

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

# Function to extract hidden data using LSB steganography
def extract_lsb(data):
    # Extract the least significant bits
    lsb_data = np.unpackbits(data.astype(np.uint8))
    hidden_data = lsb_data[::8]  # Assuming 1 bit per sample

    # Convert bits to list
    hidden_data_list = hidden_data.tolist()
    return hidden_data_list

# Function to extract hidden data using FFT
def extract_fft(data):
    # Perform FFT
    freq_data = fft(data)

    # Extract hidden data from the frequency components
    hidden_data = np.real(freq_data)  # Example: Extract real part
    hidden_data = (hidden_data - np.min(hidden_data)) / (np.max(hidden_data) - np.min(hidden_data))  # Normalize
    hidden_data = (hidden_data * 255).astype(np.uint8)  # Scale to 8-bit values

    # Convert to list
    hidden_data_list = hidden_data.tolist()
    return hidden_data_list

# Function to extract hidden data using echo hiding
def extract_echo(data, rate, delay=0.1, threshold=0.01):
    # Detect echoes
    echo_data = np.zeros_like(data)
    delay_samples = int(rate * delay)
    for i in range(delay_samples, len(data)):
        if np.abs(data[i] - data[i - delay_samples]) > threshold:
            echo_data[i] = 1

    # Convert echo data to list
    echo_data_list = echo_data.tolist()
    return echo_data_list

# Function to interpret data as ASCII characters
def interpret_as_ascii(data_list):
    # Convert list of integers to string of ASCII characters
    ascii_string = ''.join(chr(int(value)) for value in data_list if 0 <= value < 128)
    return ascii_string

# Main function to read audio file and extract hidden data
def main():
    file_path = '/home/harpoon/Documents/untitled.wav'  # Replace with your WAV file path
    rate, data = read_audio(file_path)
    
    # Resample the audio to a lower sample rate for better peak detection
    target_rate = 8000  # Target sample rate (adjust as needed)
    data, rate = resample_audio(data, rate, target_rate)
    
    # Extract hidden data using LSB steganography
    hidden_data_lsb = extract_lsb(data)
    ascii_data_lsb = interpret_as_ascii(hidden_data_lsb)
    print(f"Extracted hidden data using LSB (ASCII): {ascii_data_lsb}")
    
    # Extract hidden data using FFT steganography
    hidden_data_fft = extract_fft(data)
    ascii_data_fft = interpret_as_ascii(hidden_data_fft)
    print(f"Extracted hidden data using FFT (ASCII): {ascii_data_fft}")
    
    # Extract hidden data using echo hiding
    hidden_data_echo = extract_echo(data, rate)
    ascii_data_echo = interpret_as_ascii(hidden_data_echo)
    print(f"Extracted hidden data using Echo Hiding (ASCII): {ascii_data_echo}")
    
    # Detect and analyze peaks
    peaks = detect_peaks(data)
    analyze_peaks(peaks, data)

if __name__ == "__main__":
    main()