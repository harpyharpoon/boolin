import numpy as np
from scipy.io.wavfile import write
from PIL import Image

def extract_audio_from_image(image_path, audio_sample_rate=44100):
    # Step 1: Load the Image
    img = Image.open(image_path).convert('L')  # Grayscale
    img_data = np.array(img)
    
    # Step 2: Flatten Image Data to Form Audio Samples
    audio_data = img_data.flatten()  # Create a 1D array of pixel values
    audio_data = audio_data - 128  # Center data around 0 (optional)
    
    # Step 3: Normalize the Audio Data
    audio_data = audio_data / np.max(np.abs(audio_data))  # Normalize to range -1 to 1
    audio_data = (audio_data * 32767).astype(np.int16)  # Convert to 16-bit PCM format
    
    # Step 4: Save the Audio File
    output_audio_path = "extracted_audio.wav"
    write(output_audio_path, audio_sample_rate, audio_data)
    print(f"Audio extracted and saved to {output_audio_path}")
    return output_audio_path

# Example Usage
if __name__ == "__main__":
    image_path = "/home/harpoon/Documents/steg/041c0a8277822f0f2f65a2612061b00a-full.jpeg"  # Replace with the path to your image
    extract_audio_from_image(image_path)
