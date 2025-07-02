import numpy as np
import mrcfile
import cv2
import os
from numpy.fft import fft2, ifft2
from scipy.signal import gaussian

def transform(image):
    i_min = image.min()
    i_max = image.max()
    image = ((image - i_min) / (i_max - i_min)) * 255
    return image.astype(np.uint8)


def standard_scaler(image):
    kernel_size = 9
    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    mu = np.mean(image)
    sigma = np.std(image)
    image = (image - mu) / sigma
    image = transform(image).astype(np.uint8)
    return image


def contrast_enhancement(image):
    enhanced_image = cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)
    return enhanced_image


def gaussian_kernel(kernel_size=3):
    h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
    h = np.dot(h, h.transpose())
    h /= np.sum(h)
    return h


def wiener_filter(img, kernel, K):
    kernel /= np.sum(kernel)
    dummy = np.copy(img)
    dummy = fft2(dummy)
    kernel = fft2(kernel, s=img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.abs(ifft2(dummy))
    return dummy


def clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    img_equalized = clahe.apply(transform(image))
    return img_equalized


def guided_filter(input_image, guidance_image, radius=20, epsilon=0.1):
    input_image = input_image.astype(np.float32) / 255.0
    guidance_image = guidance_image.astype(np.float32) / 255.0

    mean_guidance = cv2.boxFilter(guidance_image, -1, (radius, radius))
    mean_input = cv2.boxFilter(input_image, -1, (radius, radius))

    mean_guidance_input = cv2.boxFilter(guidance_image * input_image, -1, (radius, radius))
    covariance_guidance_input = mean_guidance_input - mean_guidance * mean_input

    mean_guidance_sq = cv2.boxFilter(guidance_image * guidance_image, -1, (radius, radius))
    variance_guidance = mean_guidance_sq - mean_guidance * mean_guidance

    a = covariance_guidance_input / (variance_guidance + epsilon)
    b = mean_input - a * mean_guidance
    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))

    output_image = mean_a * guidance_image + mean_b
    return transform(output_image)



def denoise(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image at {image_path}")

    normalized_image = standard_scaler(image)

    clahe_image = clahe(normalized_image)

    guided_filter_image = guided_filter(clahe_image, normalized_image)

    return guided_filter_image


def process_jpg_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith(".jpg"):
            image_path = os.path.join(input_dir, file_name)
            try:
                denoised_image = denoise(image_path)

                output_path = os.path.join(output_dir, file_name)
                cv2.imwrite(output_path, denoised_image)
                print(f"Processed and saved: {output_path}")
            except ValueError as e:
                print(f"Error processing {file_name}: {e}")


input_directory = "./data/10947/denoised"
output_directory = "./data/10947/en"


process_jpg_images(input_directory, output_directory)



