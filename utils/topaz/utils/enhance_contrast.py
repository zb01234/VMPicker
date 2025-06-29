# Code for implementing different different image processing techniques for denoising.

import numpy as np
import mrcfile
import cv2
from numpy.fft import fft2, ifft2
from scipy.signal import gaussian

def transform(image):    # 归一化：实现值域映射，使输入图像的像素值范围标准化，有利于统一不同图像的亮度和对比度。
    i_min = image.min()
    i_max = image.max()

    image = ((image - i_min)/(i_max - i_min)) * 255
    return image.astype(np.uint8)


def standard_scaler(image):   # 标准化后的图像数据具有相似的亮度和对比度特性，使模型对输入数据的变化更具鲁棒性，有助于提升模型在训练和预测时的表现。
    kernel_size = 9
    # image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    mu = np.mean(image)
    sigma = np.std(image)
    image = (image - mu)/sigma
    image = transform(image).astype(np.uint8)      # astype(np.uint8)：灰度化
    return image

def contrast_enhancement(image):
    enhanced_image = cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    return enhanced_image


def gaussian_kernel(kernel_size = 3):
    h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
    h = np.dot(h, h.transpose())
    h /= np.sum(h)
    return h

def wiener_filter(img, kernel, K):
    kernel /= np.sum(kernel)
    dummy = np.copy(img)
    dummy = fft2(dummy)
    kernel = fft2(kernel, s = img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.abs(ifft2(dummy))
    return dummy

def clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))

    # Apply CLAHE to the image
    img_equalized = clahe.apply(transform(image))
    # return img_equalized
    return img_equalized.astype(np.float32) / 255.0  # 转换回 [0, 1]

def guided_filter(input_image, guidance_image, radius=20, epsilon=0.1):
    # Convert images to float32
    input_image = input_image.astype(np.float32) / 255.0
    guidance_image = guidance_image.astype(np.float32) / 255.0

    # Compute mean values of the guidance image and input image
    mean_guidance = cv2.boxFilter(guidance_image, -1, (radius, radius))
    mean_input = cv2.boxFilter(input_image, -1, (radius, radius))

    # Compute correlation and covariance of the guidance and input images
    mean_guidance_input = cv2.boxFilter(guidance_image * input_image, -1, (radius, radius))
    covariance_guidance_input = mean_guidance_input - mean_guidance * mean_input

    # Compute squared mean of the guidance image
    mean_guidance_sq = cv2.boxFilter(guidance_image * guidance_image, -1, (radius, radius))
    variance_guidance = mean_guidance_sq - mean_guidance * mean_guidance

    # Compute weights and mean of the weights
    a = covariance_guidance_input / (variance_guidance + epsilon)
    b = mean_input - a * mean_guidance
    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))

    # Compute the filtered image
    output_image = mean_a * guidance_image + mean_b

    return transform(output_image)

kernel = gaussian_kernel(kernel_size = 9)
def denoise(image_path):
    image = mrcfile.read(image_path)
    image = image.T
    image = np.rot90(image)
    normalized_image = standard_scaler(np.array(image))
    contrast_enhanced_image = contrast_enhancement(normalized_image)
    weiner_filtered_image = wiener_filter(contrast_enhanced_image, kernel, K = 30)
    clahe_image = clahe(weiner_filtered_image)
    guided_filter_image = guided_filter(clahe_image, weiner_filtered_image)
    
    return guided_filter_image

    
def denoise_jpg_image(image):
    normalized_image = standard_scaler(np.array(image))
    contrast_enhanced_image = contrast_enhancement(normalized_image)
    weiner_filtered_image = wiener_filter(contrast_enhanced_image, kernel, K = 30)
    clahe_image = clahe(weiner_filtered_image)
    guided_filter_image = guided_filter(clahe_image, weiner_filtered_image)
    
    return guided_filter_image 