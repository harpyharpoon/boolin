import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tkinter as tk
from tkinter import simpledialog

# Function to run fttmap with given coordinates
def run_fttmap(x, y, radius):
    import subprocess
    subprocess.run(['python', 'fttmap.py', str(x), str(y), str(radius)])

# Load the image
image = cv2.imread('01c9708b898d94c94fcbf970d02f7d77-full.jpeg', 0)  # Load in grayscale

# Perform FFT
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# Plot the original image
plt.figure()
plt.imshow(image, cmap='gray')
plt.title('Input Image')
plt.grid(True)
plt.show()

# Plot the magnitude spectrum
plt.figure()
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')
plt.grid(True)
plt.show()

# Create a simple GUI for coordinate input
root = tk.Tk()
root.withdraw()  # Hide the root window

x = simpledialog.askinteger("Input", "Enter X coordinate:")
y = simpledialog.askinteger("Input", "Enter Y coordinate:")
radius = simpledialog.askinteger("Input", "Enter radius:")

if x is not None and y is not None and radius is not None:
    run_fttmap(x, y, radius)