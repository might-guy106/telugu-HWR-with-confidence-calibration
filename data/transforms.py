import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
from scipy.ndimage import gaussian_filter

class ResizeWithPad:
    """
    Resize image to target height while maintaining aspect ratio and pad width.

    Args:
        target_height (int): Target height for images
        max_width (int): Maximum width after padding
        pad_value (int): Value to use for padding (default: 255 for white)
    """
    def __init__(self, target_height, max_width, pad_value=255):
        self.target_height = target_height
        self.max_width = max_width
        self.pad_value = pad_value

    def __call__(self, img):
        """Transform image by resizing and padding."""
        # Get original image dimensions
        w, h = img.size

        # Skip processing if image is invalid
        if w <= 0 or h <= 0:
            return img

        # Calculate new width to maintain aspect ratio
        new_w = int(w * (self.target_height / h))

        # Make sure new_width is at least 1 pixel
        new_w = max(1, min(new_w, self.max_width))

        # Resize image to target height maintaining aspect ratio
        img = img.resize((new_w, self.target_height), Image.LANCZOS)

        # Create padded image (pad on right side only)
        padded_img = Image.new('L', (self.max_width, self.target_height), self.pad_value)
        padded_img.paste(img, (0, 0))

        return padded_img


class ElasticTransform:
    """
    Apply elastic deformation to images as data augmentation.

    Args:
        alpha (float): Intensity of deformation
        sigma (float): Smoothness of deformation
    """
    def __init__(self, alpha=40, sigma=4):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, img):
        """Apply elastic transform to the image."""
        # Convert PIL image to numpy array
        img_np = np.array(img)
        shape = img_np.shape

        # Generate random displacement fields
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha

        # Create meshgrid for sampling
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

        # Apply displacement with bounds checking
        x_mapped = np.clip(x + dx, 0, shape[1] - 1).astype(np.float32)
        y_mapped = np.clip(y + dy, 0, shape[0] - 1).astype(np.float32)

        # Use OpenCV for remapping
        distorted_image = cv2.remap(img_np, x_mapped, y_mapped,
                                  interpolation=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=255)

        return Image.fromarray(distorted_image)


def get_transforms(phase, img_height=64, img_width=256):
    """
    Create preprocessing and augmentation pipeline.

    Args:
        phase (str): 'train', 'val', or 'test'
        img_height (int): Target image height
        img_width (int): Maximum image width

    Returns:
        transforms.Compose: Composed transformation pipeline
    """
    if phase == 'train':
        return transforms.Compose([
            # Basic preprocessing
            ResizeWithPad(img_height, img_width, pad_value=255),

            # Data augmentation (only for training)
            transforms.RandomApply([
                ElasticTransform(alpha=30, sigma=3)
            ], p=0.5),
            transforms.RandomApply([
                transforms.RandomAffine(degrees=3, shear=5,
                                       fill=255)
            ], p=0.3),
            # Optional contrast/brightness adjustments
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.3, contrast=0.3)
            ], p=0.3),

            # Convert to tensor and normalize
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])
    else:  # val or test
        return transforms.Compose([
            # Only basic preprocessing for validation/test
            ResizeWithPad(img_height, img_width, pad_value=255),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])


def get_trocr_transforms(phase, img_height=384, img_width=384):
    """
    Create preprocessing and augmentation pipeline for TrOCR models.

    Args:
        phase (str): 'train', 'val', or 'test'
        img_height (int): Target image height
        img_width (int): Maximum image width

    Returns:
        transforms.Compose: Composed transformation pipeline
    """
    if phase == 'train':
        return transforms.Compose([
            # Basic preprocessing
            ResizeWithPad(img_height, img_width, pad_value=255),

            # Data augmentation (only for training)
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.3, contrast=0.3)
            ], p=0.3),

            # Convert to tensor - don't normalize here since TrOCR processor will do that
            transforms.ToTensor(),

            # The processor expects values in [0, 255], but ToTensor scales to [0, 1]
            # We'll scale back to [0, 255] in the _preprocess_images method
        ])
    else:  # val or test
        return transforms.Compose([
            # Only basic preprocessing for validation/test
            ResizeWithPad(img_height, img_width, pad_value=255),
            transforms.ToTensor(),
        ])
