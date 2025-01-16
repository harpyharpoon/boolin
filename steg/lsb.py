import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import subprocess

def extract_lsb(image_path):
    """Extract Least Significant Bits (LSBs) from an image."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print("Error: Could not load the image.")
        return None
    lsb_image = img & 1  # Extract LSB by bitwise AND
    return lsb_image

def perform_fft(image_path):
    """Perform FFT on a grayscale image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Could not load the image.")
        return None
    f_transform = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    return magnitude_spectrum

def extract_metadata(image_path):
    """Extract metadata from an image using ExifTool."""
    try:
        process = subprocess.Popen(
            ["exiftool", image_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        stdout, _ = process.communicate()
        return stdout
    except FileNotFoundError:
        print("Error: ExifTool is not installed.")
        return None

def analyze_color_channels(image_path):
    """Analyze each color channel for unusual patterns."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print("Error: Could not load the image.")
        return None
    channels = cv2.split(img)
    return channels

def display_image(image, title="Image"):
    """Display an image."""
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def main(image_path):
    print("Running steganography tests on:", image_path)
    
    # LSB Extraction
    print("Extracting Least Significant Bits (LSB)...")
    lsb_image = extract_lsb(image_path)
    if lsb_image is not None:
        display_image(lsb_image, "LSB Image")

    # FFT Analysis
    print("Performing FFT analysis...")
    fft_image = perform_fft(image_path)
    if fft_image is not None:
        display_image(fft_image, "FFT Magnitude Spectrum")

    # Metadata Extraction
    print("Extracting metadata...")
    metadata = extract_metadata(image_path)
    if metadata:
        print("Metadata:\n", metadata)

    # Color Channel Analysis
    print("Analyzing color channels...")
    channels = analyze_color_channels(image_path)
    if channels:
        for i, channel in enumerate(channels):
            display_image(channel, f"Channel {i}")

# Path to the image
image_path = "1736617941944326.png"  # Replace with your image path
main(image_path)
