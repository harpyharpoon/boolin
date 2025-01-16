import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

# Morse code dictionary
MORSE_CODE_DICT = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E', '..-.': 'F',
    '--.': 'G', '....': 'H', '..': 'I', '.---': 'J', '-.-': 'K', '.-..': 'L',
    '--': 'M', '-.': 'N', '---': 'O', '.--.': 'P', '--.-': 'Q', '.-.': 'R',
    '...': 'S', '-': 'T', '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X',
    '-.--': 'Y', '--..': 'Z', '-----': '0', '.----': '1', '..---': '2',
    '...--': '3', '....-': '4', '.....': '5', '-....': '6', '--...': '7',
    '---..': '8', '----.': '9'
}

# Function to decode Morse code
def decode_morse(morse_code):
    return ''.join(MORSE_CODE_DICT.get(code, '') for code in morse_code.split(' '))

# Function to process audio and decode Morse code
def process_audio(data, rate, threshold=0.01):
    morse_code = ''
    signal = data / np.max(np.abs(data))  # Normalize
    signal = np.where(signal > threshold, 1, 0)  # Binarize
    morse_code += decode_signal(signal, rate)
    print(f"Decoded Morse Code: {morse_code}")
    print(f"Decoded Text: {decode_morse(morse_code)}")

# Function to decode the signal into Morse code
def decode_signal(signal, rate):
    morse_code = ''
    unit_time = rate // 10  # Assuming 10 units per second
    for i in range(0, len(signal), unit_time):
        segment = signal[i:i + unit_time]
        if np.mean(segment) > 0.5:
            morse_code += '-'
        else:
            morse_code += '.'
    return morse_code

# Main function to read audio file and decode Morse code
def main():
    file_path = '/home/harpoon/Documents/untitled.wav'  # Replace with your WAV file path
    rate, data = wavfile.read(file_path)
    
    # Handle stereo by selecting one channel if necessary
    if len(data.shape) > 1:
        data = data[:, 0]
    
    print("Processing audio file for Morse code...")
    process_audio(data, rate)

if __name__ == "__main__":
    main()