import numpy as np
import pywt
from PIL import Image
import math

# Step 1: Perform DWT on the Stego Image
def apply_dwt(image):
    coeffs2 = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs2
    return LL, LH, HL, HH

# Step 2: Extract 3-bit Blocks from the Selected Sub-band
def extract_data(sub_band, num_blocks):
    flat_band = sub_band.flatten()
    data_blocks = [int(flat_band[i] % 8) for i in range(num_blocks)]
    return data_blocks

# Step 3: Reconstruct Huffman Encoded Data
def reconstruct_huffman(data_blocks):
    print("Data blocks:", data_blocks)  # Debug statement
    binary_secret = ''.join(format(block, '03b') for block in data_blocks)
    print("Binary secret length:", len(binary_secret))  # Debug statement
    return binary_secret

# Step 4: Convert Huffman Data Back to an Image
def huffman_decode(binary_secret, original_shape):
    # Ensure the binary secret length is a multiple of 8
    if len(binary_secret) % 8 != 0:
        binary_secret = binary_secret[:-(len(binary_secret) % 8)]
    
    pixels = [
        int(binary_secret[i:i+8], 2) for i in range(0, len(binary_secret), 8)
    ]
    print("Number of pixels:", len(pixels))  # Debug statement
    
    # Ensure the number of pixels matches the expected size
    expected_size = original_shape[0] * original_shape[1]
    if len(pixels) < expected_size:
        raise ValueError(f"Not enough data to reconstruct the image. Expected {expected_size} pixels, got {len(pixels)}.")
    
    decoded_image = np.array(pixels[:expected_size], dtype=np.uint8).reshape(original_shape)
    return decoded_image

# Function to estimate the image size
def estimate_image_size(num_pixels):
    # Find the closest factors of num_pixels that form a reasonable image dimension
    sqrt_val = int(math.sqrt(num_pixels))
    for i in range(sqrt_val, 0, -1):
        if num_pixels % i == 0:
            return (i, num_pixels // i)
    return (num_pixels, 1)

# Main Function
def decode_secret(stego_path, num_blocks):
    stego_image = Image.open(stego_path).convert('L')  # Load stego image
    stego_image = np.array(stego_image)
    
    LL, LH, HL, HH = apply_dwt(stego_image)
    
    # Extract data blocks from all sub-bands
    data_blocks = []
    data_blocks.extend(extract_data(LL, num_blocks))
    data_blocks.extend(extract_data(LH, num_blocks))
    data_blocks.extend(extract_data(HL, num_blocks))
    data_blocks.extend(extract_data(HH, num_blocks))
    
    # Reconstruct Huffman-encoded binary data
    binary_secret = reconstruct_huffman(data_blocks)
    
    # Estimate the image size
    num_pixels = len(binary_secret) // 8
    estimated_shape = estimate_image_size(num_pixels)
    print(f"Estimated image shape: {estimated_shape}")
    
    # Decode binary data back into the secret image
    secret_image = huffman_decode(binary_secret, estimated_shape)
    return secret_image

# Example Usage
if __name__ == "__main__":
    stego_path = "/home/harpoon/Documents/steg/041c0a8277822f0f2f65a2612061b00a-full.jpeg"  # Replace with your stego image path
    num_blocks = 10000  # Define the number of blocks to extract from each sub-band
    secret_image = decode_secret(stego_path, num_blocks)
    Image.fromarray(secret_image).show()
