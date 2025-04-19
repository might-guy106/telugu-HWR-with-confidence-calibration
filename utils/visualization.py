import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(ground_truth, predictions, labels, output_file):
    """
    Plot confusion matrix for character predictions.

    Args:
        ground_truth: List of ground truth characters
        predictions: List of predicted characters
        labels: List of character labels
        output_file: Path to save the plot
    """
    cm = confusion_matrix(ground_truth, predictions, labels=labels)

    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized[np.isnan(cm_normalized)] = 0

    plt.rcParams['font.family'] = 'Noto Sans Telugu, sans-serif'

    # Plot
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Character Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_error_distribution(ground_truth, predictions, output_file):
    """
    Plot distribution of error lengths.

    Args:
        ground_truth: List of ground truth strings
        predictions: List of predicted strings
        output_file: Path to save the plot
    """
    from utils.metrics import levenshtein_distance

    error_lengths = []

    for gt, pred in zip(ground_truth, predictions):
        distance = levenshtein_distance(gt, pred)
        error_lengths.append(distance)

    plt.figure(figsize=(10, 6))
    plt.hist(error_lengths, bins=20, color='blue', alpha=0.7, edgecolor='black')
    plt.title('Distribution of Error Lengths')
    plt.xlabel('Levenshtein Distance')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_training_history(history, output_file):
    """
    Plot training history metrics.

    Args:
        history: Dictionary containing training metrics
        output_file: Path to save the plot
    """
    plt.figure(figsize=(15, 10))

    # Plot training and validation loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot character error rate
    plt.subplot(2, 2, 2)
    plt.plot(history['val_cer'], label='CER')
    plt.title('Character Error Rate')
    plt.xlabel('Epoch')
    plt.ylabel('CER (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot word error rate
    plt.subplot(2, 2, 3)
    plt.plot(history['val_wer'], label='WER')
    plt.title('Word Error Rate')
    plt.xlabel('Epoch')
    plt.ylabel('WER (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot learning rate
    if 'learning_rates' in history:
        plt.subplot(2, 2, 4)
        plt.plot(history['learning_rates'], label='Learning Rate')
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
