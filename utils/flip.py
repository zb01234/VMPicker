import cv2
import os
import glob

IMAGE_DIRECTORY = './Evaluation/General/VMPicker/10081/masks'

image_files = glob.glob(os.path.join(IMAGE_DIRECTORY, '*.jpg'))

for image_file in image_files:
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Error reading image: {image_file}")
        continue
    
    flipped_image = cv2.flip(image, 0)
    
    cv2.imwrite(image_file, flipped_image)
    print(f"Successfully flipped and saved: {image_file}")

print("All images have been flipped and saved.")

