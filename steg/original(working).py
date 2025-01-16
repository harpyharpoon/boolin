import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift

def show_image(img, title="Image"):
    """
    Display an image using matplotlib.

    Parameters:
    img (ndarray): The image to be displayed.
    title (str): The title of the image window.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()

def apply_fourier_transform(image):
    """
    Apply Fourier Transform to the input image and return the magnitude spectrum and the Fourier transformed image.

    Parameters:
    image (numpy.ndarray): Input grayscale image.

    Returns:
    tuple: A tuple containing the magnitude spectrum (numpy.ndarray) and the Fourier transformed image (numpy.ndarray).
    """
    f_transform = fft2(image)
    f_shifted = fftshift(f_transform)  # Shift the zero frequency component to the center
    magnitude_spectrum = np.abs(f_shifted)
    return magnitude_spectrum, f_transform

def apply_bandpass_filter(f_transform, low_cutoff=30, high_cutoff=150):
    """
    Apply a bandpass filter to the Fourier transform of an image.

    Parameters:
    f_transform (ndarray): The Fourier transform of the image.
    low_cutoff (int): The lower cutoff frequency for the bandpass filter.
    high_cutoff (int): The higher cutoff frequency for the bandpass filter.

    Returns:
    ndarray: The filtered Fourier transform.
    """
    rows, cols = f_transform.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), dtype=bool)
    mask[crow-low_cutoff:crow+low_cutoff, ccol-low_cutoff:ccol+low_cutoff] = False
    mask[crow-high_cutoff:crow+high_cutoff, ccol-high_cutoff:ccol+high_cutoff] = True
    f_transform[mask] = 0
    return f_transform

def edge_detection(image):
    """
    Perform edge detection on the input image using the Canny algorithm.

    Parameters:
    image (numpy.ndarray): Input grayscale image.

    Returns:
    numpy.ndarray: Image with edges detected.
    """
    edges = cv2.Canny(image, 100, 200)
    return edges

def main(image_path):
    """
    Main function to process the image.

    Parameters:
    image_path (str): The path to the image file to be processed.
    """
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Show the original image
    show_image(image, title="Original Image")

    # Apply Fourier Transform and show magnitude spectrum
    magnitude_spectrum, f_transform = apply_fourier_transform(image)
    show_image(np.log(magnitude_spectrum + 1), title="Magnitude Spectrum (Log Scale)")

    # Apply a simple bandpass filter to remove low and high frequencies
    filtered_f_transform = apply_bandpass_filter(f_transform.copy())
    filtered_image = np.abs(ifft2(filtered_f_transform))
    show_image(filtered_image, title="Filtered Image (Bandpass)")

    # Apply edge detection
    edges = edge_detection(image)
    show_image(edges, title="Edge Detection (Canny)")

if __name__ == "__main__":
    # Hardcode the image path here for testing
    image_path = "/home/harpoon/Documents/steg/idkdump.png"
    main(image_path)
