import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class TeluguHWRDataset(Dataset):
    """
    Dataset for Telugu Handwritten Word Recognition

    This dataset loads image-text pairs from a text file and applies
    transformations to the images. The text file should have one
    sample per line with format: 'image_path label'.

    Args:
        data_file (str): Path to text file with annotations
        root_dir (str): Root directory containing the dataset
        transform (callable, optional): Optional transform to be applied to images
        max_samples (int, optional): Maximum number of samples to load (None for all)
    """
    def __init__(self, data_file, root_dir, transform=None, max_samples=None):
        """
        Initialize the dataset.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Read the data file and extract image paths and labels
        logger.info(f"Loading data from {data_file}")
        with open(data_file, 'r', encoding='utf-8') as f:
            sample_count = 0
            for line in f:
                if max_samples is not None and sample_count >= max_samples:
                    break

                parts = line.strip().split(' ', 1)  # Split only on the first space
                if len(parts) == 2:
                    img_path = parts[0]
                    # Join all remaining parts as the label (may contain spaces)
                    label = parts[1]
                    self.samples.append((img_path, label))
                    sample_count += 1
                else:
                    logger.info(f"Skipping invalid line: {line.strip()}")

        logger.info(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Return image and label at the given index."""
        img_path, label = self.samples[idx]
        img_path = os.path.join(self.root_dir, img_path)

        try:
            image = Image.open(img_path).convert('L')  # Convert to grayscale

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a placeholder image and label in case of error
            placeholder = Image.new('L', (100, 32), color=255)
            if self.transform:
                placeholder = self.transform(placeholder)
            return placeholder, "error"
