import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift
import tkinter as tk
from tkinter import simpledialog

def create_circular_mask(h, w, center=None, radius=None):
    """
    Create a circular mask.

    Parameters:
    h (int): Height of the mask.
    w (int): Width of the mask.
    center (tuple): Center of the circle (x, y). Defaults to the center of the image.
    radius (int): Radius of the circle. Defaults to the smallest distance to the image edge.

    Returns:
    np.ndarray: Circular mask.
    """
    if center is None:
        center = (int(w / 2), int(h / 2))
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = np.zeros((h, w), dtype=np.float32)
    mask[dist_from_center <= radius] = 1
    return mask

def apply_mask(fshift, mask):
    """
    Apply a mask to the FFT.

    Parameters:
    fshift (np.ndarray): Shifted FFT of the image.
    mask (np.ndarray): Mask to apply.

    Returns:
    np.ndarray: Masked FFT.
    """
    return fshift * mask

def show_image_with_grid(image, title):
    """
    Display an image in a separate window with a grid.

    Parameters:
    image (np.ndarray): Image to display.
    title (str): Title of the window.
    """
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.grid(True)
    plt.show()

def apply_fourier_transform(image):
    """
    Apply Fourier Transform to an image.

    Parameters:
    image (np.ndarray): Input image.

    Returns:
    tuple: Magnitude spectrum and Fourier Transform of the image.
    """
    f_transform = fft2(image)
    f_shifted = fftshift(f_transform)  # Shift the zero frequency component to the center
    magnitude_spectrum = np.abs(f_shifted)
    return magnitude_spectrum, f_transform

def main():
    """
    Main processing function.
    """
    # Create a simple GUI for file selection and coordinate input
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Get image file path
    image_path = simpledialog.askstring("Input", "Enter the path to the image file:")
    if not image_path:
        print("No image file path provided.")
        return

    # Load the image
    image = cv2.imread(image_path, 0)  # Load in grayscale
    if image is None:
        print(f"Failed to load image from {image_path}")
        return

    # Apply Fourier Transform
    magnitude_spectrum, f_transform = apply_fourier_transform(image)

    # Display images
    show_image_with_grid(image, "Input Image")
    show_image_with_grid(magnitude_spectrum, "Magnitude Spectrum")

    # Get coordinates and radius
    x = simpledialog.askinteger("Input", "Enter X coordinate:")
    y = simpledialog.askinteger("Input", "Enter Y coordinate:")
    radius = simpledialog.askinteger("Input", "Enter radius:")
    if x is None or y is None or radius is None:
        print("Invalid coordinates or radius.")
        return

    # Create and apply mask
    mask = create_circular_mask(image.shape[0], image.shape[1], center=(x, y), radius=radius)
    f_transform_masked = apply_mask(f_transform, mask)

    # Inverse FFT to get the image back
    f_ishift = np.fft.ifftshift(f_transform_masked)
    img_back = ifft2(f_ishift)
    img_back = np.abs(img_back)

    # Display masked image and mask
    show_image_with_grid(mask, "Mask")
    show_image_with_grid(img_back, "Masked Image")

if __name__ == "__main__":
    main()
