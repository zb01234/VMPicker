import os
import numpy as np
import mrcfile
import cv2
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
    return transform(image).astype(np.uint8)

def contrast_enhancement(image):
    enhanced_image = cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)
    return enhanced_image

def gaussian_kernel(kernel_size=3):
    h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
    h = np.dot(h, h.T)
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

    dummy = (dummy - dummy.min()) / (dummy.max() - dummy.min()) * 255
    return dummy.astype(np.uint8)

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

def denoise_and_save(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    kernel = gaussian_kernel(kernel_size=9)
    for filename in os.listdir(folder_path):
        if filename.endswith(".mrc"):
            image_path = os.path.join(folder_path, filename)
            with mrcfile.open(image_path, permissive=True) as mrc:
                image = mrc.data.T
            image = np.rot90(image)

            # Step 1: Standard Scaler
            std_scaled_image = standard_scaler(np.array(image))
            cv2.imwrite(os.path.join(output_folder, f"{filename}_standard_scaler.jpg"), std_scaled_image)

            # Step 2: Contrast Enhancement
            contrast_enhanced_image = contrast_enhancement(std_scaled_image)
            cv2.imwrite(os.path.join(output_folder, f"{filename}_contrast_enhancement.jpg"), contrast_enhanced_image)

            # Step 3: Wiener Filter
            weiner_filtered_image = wiener_filter(contrast_enhanced_image, kernel, K=30)
            cv2.imwrite(os.path.join(output_folder, f"{filename}_wiener_filter.jpg"), weiner_filtered_image.astype(np.uint8))

            # Step 4: CLAHE
            clahe_image = clahe(weiner_filtered_image)
            cv2.imwrite(os.path.join(output_folder, f"{filename}_clahe.jpg"), clahe_image)

            # Step 5: Guided Filter
            guided_filtered_image = guided_filter(clahe_image, weiner_filtered_image)
            cv2.imwrite(os.path.join(output_folder, f"{filename}_guided_filter.jpg"), guided_filtered_image)

            # Final processed image after all steps
            cv2.imwrite(os.path.join(output_folder, f"{filename}_final_processed.jpg"), guided_filtered_image)
