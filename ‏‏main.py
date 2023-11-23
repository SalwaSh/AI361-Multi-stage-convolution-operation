"""
Name: Salwa Shama
OS: Windows 11
Python 3.10.9

This is a small program that provides the following functionalities: Executing a three-stage convolution operation with an 
edge detection filter on gray images, preceded by noise addition and reduction techniques
"""
import cv2 
import os
import numpy as np
from image_utilities import load_image, enhance_image, add_salt_and_pepper_noise, remove_salt_and_pepper_noise, convolve2D

# Define the path for the input folder (to read images) and output folder (to save the images).
input_folder = "path_folder " # Provide your folder path here for input folder
output_folder = "path_folder" # Provide your folder path here for output folder

# Get the list of all images in the input folder.
folder = os.listdir(input_folder)

# Iterate through each image in the folder.
for image in folder:
    if image.lower().endswith(('.png')):
        image_path = os.path.join(input_folder, image)

        # Generate output image paths with different suffixes names (to save different versions of edge detection).
        output_name = image.split('.')[0]
        output_path_a = os.path.join(output_folder, output_name + '_A.png')
        output_path_b = os.path.join(output_folder, output_name + '_B.png')
        output_path_c = os.path.join(output_folder, output_name + '_C.png')

        # Load the image.
        image = load_image(image_path)
        # Enhance the image (preprocess step).
        enhanced_image = enhance_image(image)

        # Add salt-and-pepper noise to the image.
        noisy_image = add_salt_and_pepper_noise(enhanced_image)
        # Remove salt-and-pepper noise from the image using median filter.
        filter_size = 3
        denoised_image = remove_salt_and_pepper_noise(noisy_image, filter_size)

        # Define a Sobel filter for convolution (horizontal edge detection).
        Sobel_filter = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

        # Apply convolution in three stages.
        strides = 2
        result = convolve2D(denoised_image, Sobel_filter, strides)
        cv2.imwrite(output_path_a, result)
        
        strides = 1
        result2 = convolve2D(result, Sobel_filter, strides)
        cv2.imwrite(output_path_b, result2)
        
        strides = 1
        result3 = convolve2D(result2, Sobel_filter, strides)
        cv2.imwrite(output_path_c, result3)

        # Print a message indicating completion of current image.
        print("Done from", image_path, "✅")
