import cv2 
from skimage import util
from skimage.util import random_noise
import numpy as np

def load_image(image_path): 
    """
    Load an image using OpenCV and converts it to grayscale.
    Arguments:
        image_path (str): The path to the input image.
    Returns:
        gray_image (numpy.ndarray): The grayscale image.
    Exceptions:
        None.
    """
    # Load the image.
    image = cv2.imread(image_path) 
    # Convert the image to grayscale.
    gray_image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY) 
    return gray_image

def enhance_image(image):
    """
    Enhance the input image using histogram equalization.
    Arguments:
        image (numpy.ndarray): The grayscale image to be enhanced.
    Returns:
        image_equalize (numpy.ndarray): The enhanced grayscale image after applying histogram equalization.
    Exceptions:
        None.
    """
    # Applies histogram equalization.
    image_equalize = cv2.equalizeHist(image)
    return image_equalize

# Add_salt_and_pepper_nois function
# Adapted from: soufanom, NoiseModels, (2023), ai361-ip, https://github.com/soufanom/ai361-ip/blob/main/NoiseModels.py

def add_salt_and_pepper_noise(image, salt_vs_pepper=0.5, amount=0.01):
    """
    Add salt-and-pepper noise to the input image using the random_noise function from skimage.
    Arguments:
        image (numpy.ndarray): The grayscale image to be degraded.
        salt_vs_pepper (float): The ratio of salt (white pixels) to pepper (black pixels). Should be in the range [0, 1]
        amount (float): The proportion of pixels from the entire image to be affected by noise. Should be in the range [0, 1]
    Returns:
        noisy_image_uint8 (numpy.ndarray): The grayscale image with salt-and-pepper noise added.
    Exceptions:
        None.
    """
    # Scale the image values between 0 and 1.
    image = util.img_as_float(image) 
    # Add salt-and-pepper noise using built-in function.
    noisy_image = random_noise(image, mode='s&p', salt_vs_pepper=salt_vs_pepper, amount=amount)
    # Convert the noisy image back to uint8 format.
    noisy_image_uint8 = util.img_as_ubyte(noisy_image)  
    return noisy_image_uint8

def __calculate_median(array):
    """
    Calculate the median of a set of numbers, whether the array length is even or odd.
    Arguments:
        array (numpy.ndarray): The input array for which the median will be calculated.
    Returns:
        median (integer): The median value of the input array.
    Exceptions:
        None.
    """
    # Sort the elements in the array using built-in function.
    sorted_array = sorted(array.flatten().tolist())
    # Calculate the length of the array.
    array_length = len(sorted_array)
    if array_length % 2 == 0:
        # If the array length is even, average the middle two elements
        mid_index = array_length // 2
        median = (sorted_array[mid_index - 1] + sorted_array[mid_index]) / 2
    else:
        # If the array length is odd, return the middle element
        mid_index = array_length // 2
        median = sorted_array[mid_index]
    return int(median)

def remove_salt_and_pepper_noise(image, filter_size):
    """
    Remove salt-and-pepper noise from the grayscale image using a median filter.
    Arguments:
        image (numpy.ndarray): The grayscale image containing salt-and-pepper noise.
        filter_size (int): The size of the square filter window for the median filter. Should be an odd number.
    Returns:
        output (numpy.ndarray): The grayscale image with salt-and-pepper noise removed.
    Exceptions:
        None.
    """
    # Create a square kernel of ones (to save the values of pixels in the image) with the specified filter size.
    filter = np.ones((filter_size, filter_size))
    # Set the strides to 1 (as it is a special case for the convolution operation).
    STRIDES = 1 

    # Get image dimensions.
    image_height, image_width = image.shape
    # Get filter dimensions.
    filter_height, filter_width = filter.shape

    # Calculate pad dimensions to maintain the output size.
    pad_height = int((filter_height - 1) / 2)
    pad_width = int((filter_width - 1) / 2)
    # Calculate the shape of the output convolution
    output_height = int(((image_height - filter_height + 2 * pad_height) / STRIDES) + 1)
    output_width = int(((image_width - filter_width + 2 * pad_width) / STRIDES) + 1)
    # Initialize the output array with zeros.
    output = np.zeros((output_height, output_width)) # Considering the padding and strides.

    # Apply equal padding to all sides of the image.
    image_padded = np.zeros((image.shape[0] + pad_height*2, image.shape[1] + pad_width*2)) # padding only
    # Fill the imagePadded with the pixels of the input image.
    image_padded[int(pad_height):int(-1 * pad_height), int(pad_width):int(-1 * pad_width)] = image

    # Iterate through the image to calculate the median value for the neighborhood pixels.
    for y in range(0, image_width, STRIDES):
        for x in range(0, image_height, STRIDES):
            # Extract the region of interest and apply the median filter.
            output[x // STRIDES, y // STRIDES] = __calculate_median(filter * image_padded[x: x + filter_height, y: y + filter_width])
    return output


# Convolve2D function 
# Adapted from: S. Sahoo, “2D Convolution using Python & NumPy,” Analytics Vidhya, Jan. 07, 2022. 
# https://medium.com/analytics-vidhya/2d-convolution-using-python-numpy-43442ff5f381#:~:text=To%20start%20the%202D%20Convolution 
# (accessed Nov. 16, 2023).

def convolve2D(image, filter, strides=1):
    """
    Apply 2D convolution to the grayscale image using the specified filter and strides.
    Arguments:
        image (numpy.ndarray): The grayscale image to be convolved.
        filter (numpy.ndarray): The square filter with predefined values to detect features such as edges. Should be odd size.
        strides (int): Positive integer specifying the step size or movement while applying the convolution.
    Returns:
        output (numpy.ndarray): The grayscale image after applying the convolution with the specified kernel.
    Exceptions:
        None.
    """
    # Get image dimensions.
    image_height, image_width = image.shape
    # Get filter dimensions.
    filter_height, filter_width = filter.shape

    # Calculate pad dimensions to maintain output size.
    pad_height = int((filter_height - 1) / 2)
    pad_width = int((filter_width - 1) / 2)
    # Calculate the shape of the output convolution.
    output_height = int(((image_height - filter_height + 2 * pad_height) / strides) + 1)
    output_width = int(((image_width - filter_width + 2 * pad_width) / strides) + 1)
    # Initialize the output array with zeros.
    output = np.zeros((output_height, output_width))  # Considering the padding and strides.

    # Apply equal padding to all sides of the image.
    image_padded = np.zeros((image.shape[0] + pad_height*2, image.shape[1] + pad_width*2)) # padding only
    # Fill the imagePadded with the pixels of the input image.
    image_padded[int(pad_height):int(-1 * pad_height), int(pad_width):int(-1 * pad_width)] = image

    # Iterate through the image to apply the filter.
    for y in range(0, image_width, strides):
        for x in range(0, image_height, strides):
            # Apply the convolution operation by multiplying the filter with the image region.
            sum = np.sum(filter * image_padded[x: x + filter_height, y: y + filter_width])
            # Clip the image if the pixel value is out of range.
            if sum > 255:
                sum = 255
            elif sum < 0:
                sum = 0
            output[x // strides, y // strides]= sum
    return output
