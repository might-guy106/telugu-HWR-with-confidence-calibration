import os
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter
import torch
from PIL import Image

def analyze_dataset(dataset, save_dir='.'):
    """
    Analyze dataset characteristics and visualize samples.

    Args:
        dataset: TeluguHWRDataset object
        save_dir: Directory to save visualizations

    Returns:
        vocab: Set of unique characters in the dataset
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Collect statistics
    lengths = []
    char_freq = {}
    word_counts = Counter()

    for _, label in dataset:
        # Skip error samples
        if label == "error":
            continue

        # Analyze label length
        lengths.append(len(label))

        # Analyze character frequencies
        for char in label:
            char_freq[char] = char_freq.get(char, 0) + 1

        # Count word occurrences
        word_counts[label] += 1

    # Basic statistics
    print(f"Total samples: {len(dataset)}")
    print(f"Average label length: {np.mean(lengths):.2f} chars")
    print(f"Min label length: {min(lengths)} chars")
    print(f"Max label length: {max(lengths)} chars")
    print(f"Total unique characters: {len(char_freq)}")
    print(f"Total unique words: {len(word_counts)}")

    # Calculate repeating words
    repeats = sum(1 for count in word_counts.values() if count > 1)
    repeat_percentage = (repeats / len(word_counts)) * 100 if word_counts else 0
    print(f"Words that appear multiple times: {repeats} ({repeat_percentage:.2f}%)")

    plt.rcParams['font.family'] = 'Noto Sans Telugu, sans-serif'

    # Plot distribution of label lengths
    plt.figure(figsize=(10, 5))
    plt.hist(lengths, bins=range(min(lengths), max(lengths) + 2), alpha=0.7, color='blue', edgecolor='black')
    plt.title('Distribution of Label Lengths')
    plt.xlabel('Number of Characters')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.savefig(os.path.join(save_dir, 'label_lengths_distribution.png'))
    plt.close()

    # Plot most common characters
    most_common_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)[:40]
    chars, freqs = zip(*most_common_chars)

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(chars)), freqs, color='green', alpha=0.7)
    plt.xticks(range(len(chars)), chars, rotation=45)
    plt.title('Most Common Characters in Dataset')
    plt.xlabel('Character')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'character_frequency.png'))
    plt.close()

    # Return vocabulary (set of unique characters)
    return set(char_freq.keys())


def visualize_samples(dataset, num_samples=10, cols=5, save_path='sample_images.png'):
    """
    Visualize random samples from the dataset.

    Args:
        dataset: TeluguHWRDataset object
        num_samples: Number of samples to visualize
        cols: Number of columns in the grid
        save_path: Path to save the visualization
    """
    # Get random indices
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    # Calculate rows needed
    rows = (num_samples + cols - 1) // cols

    # Set a font that supports Telugu
    plt.rcParams['font.family'] = 'Noto Sans Telugu, sans-serif'

    plt.figure(figsize=(15, rows * 3))

    for i, idx in enumerate(indices):
        image, label = dataset[idx]

        # Handle different types of image formats
        if isinstance(image, torch.Tensor):
            # Convert tensor to numpy for display
            if image.ndim == 3:  # C,H,W format
                image_np = image.permute(1, 2, 0).numpy()
                if image_np.shape[2] == 1:  # Single channel
                    image_np = image_np.squeeze(2)
            else:
                image_np = image.numpy()

            # Normalize to [0, 255] for display
            if image_np.min() < 0 or image_np.max() > 1:
                image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
            image_np = (image_np * 255).astype(np.uint8)
        else:
            # Assume PIL image
            image_np = np.array(image)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(image_np, cmap='gray')

        # Add a title with both Telugu text and code points as fallback
        title_text = f"Label: {label}"
        plt.title(f"{title_text}", fontsize=9)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Sample visualization saved to {save_path}")
