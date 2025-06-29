# Code for creating dataset

from torch.utils.data import Dataset
import mrcfile
import cv2
import torch
import config
import random
import numpy as np
from scipy import ndimage

def min_max(image):
    i_min = image.min()
    i_max = image.max()

    image = ((image - i_min)/(i_max - i_min))
    return image

def transform(image):
    i_min = image.min()
    i_max = image.max()
    
    if i_max == 0:
        return image

    image = ((image - i_min)/(i_max - i_min)) * 255
    return image.astype('uint8')


class CryoEMDataset(Dataset):
    def __init__(self, img_dir, augment=False):
        super().__init__()
        self.img_dir = img_dir
        self.augment = augment

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        image_path = self.img_dir[idx]
        mask_path = image_path[:-4] + '_mask.jpg'
        mask_path = mask_path.replace('images', 'masks')
        
        image = cv2.imread(image_path, 0)
        mask = cv2.imread(mask_path, 0)
        
        image = cv2.resize(image, (config.input_image_width, config.input_image_height))
        mask = cv2.resize(mask, (config.input_image_width, config.input_image_height))

        if self.augment:
            if random.random() > 0.5:
                image, mask = self.random_rot_flip(image, mask)
            if random.random() > 0.5:
                image, mask = self.random_rotate(image, mask)

        image = torch.from_numpy(image).unsqueeze(0).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        image = image / 255.0
        mask = mask / 255.0

        return (image, mask)

    def random_rot_flip(self, image, label):
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return image, label

    def random_rotate(self, image, label):
        angle = np.random.randint(-45, 45)
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False)
        return image, label
