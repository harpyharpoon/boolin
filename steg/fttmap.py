import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys

def apply_mask(fshift, mask):
    # Apply the mask to the shifted FFT
    fshift_masked = fshift * mask
    return fshift_masked

def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    mask = dist_from_center <= radius
    return mask

# Get coordinates and radius from command-line arguments
x = int(sys.argv[1])
y = int(sys.argv[2])
radius = int(sys.argv[3])

# Load the image
image = cv2.imread('01c9708b898d94c94fcbf970d02f7d77-full.jpeg', 0)  # Load in grayscale

# Perform FFT
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)

# Create a mask
rows, cols = image.shape
mask = create_circular_mask(rows, cols, center=(x, y), radius=radius)

# Apply the mask
fshift_masked = apply_mask(fshift, mask)

# Inverse FFT to get the image back
f_ishift = np.fft.ifftshift(fshift_masked)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# Plot the original image
plt.figure()
plt.imshow(image, cmap='gray')
plt.title('Input Image')
plt.xticks([]), plt.yticks([])

# Plot the magnitude spectrum
magnitude_spectrum = 20 * np.log(np.abs(fshift))
plt.figure()
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')
plt.xticks([]), plt.yticks([])

# Plot the mask
plt.figure()
plt.imshow(mask, cmap='gray')
plt.title('Mask')
plt.xticks([]), plt.yticks([])

# Plot the masked image
plt.figure()
plt.imshow(img_back, cmap='gray')
plt.title('Masked Image')
plt.xticks([]), plt.yticks([])

plt.show()