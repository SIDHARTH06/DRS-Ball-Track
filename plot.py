import cv2
import numpy as np

def plot_trajectory_as_mask(image,predicted_trajectory):
    """Plots the trajectory of a ball as a mask over an image using OpenCV."""
    # Create a black and white mask image with the same size as the original image
    mask = np.zeros_like(image)

    # Draw the predicted trajectory on the mask image
    for i in range(len(predicted_trajectory) - 1):
        cv2.line(mask, tuple(predicted_trajectory[i]), tuple(predicted_trajectory[i+1]), 255, 2)

    # Combine the original image with the mask image using alpha blending
    blended_image = cv2.addWeighted(image[:,:,::-1], 0.7, mask, 0.3, 0)
    return blended_image
