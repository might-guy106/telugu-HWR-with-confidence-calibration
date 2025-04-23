import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import argparse
import torch
import json
import logging
import numpy as np
from torch.utils.data import DataLoader

from data.dataset import TeluguHWRDataset
from data.transforms import get_transforms
from models.crnn import CRNN
from models.mc_dropout_crnn import MCDropoutCRNN
from models.parseq import PARSeq
from utils.ctc_decoder import CTCLabelConverter
from utils.metrics import calculate_metrics
from utils.visualization import plot_confusion_matrix, plot_error_distribution

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_vocabulary(vocab_file):
    """Load vocabulary from file"""
    vocab = set()
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
            char = line.strip()
            if char:  # Skip empty lines
                vocab.add(char)
    return vocab

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    logger.info(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load vocabulary
    vocab = load_vocabulary(args.vocab_file)
    logger.info(f"Loaded vocabulary with {len(vocab)} characters")

    # Create converter
    converter = CTCLabelConverter(vocab)

    # Create model based on the model type
    if args.model_type == 'crnn':
        if args.mc_dropout:
            model = MCDropoutCRNN(
                img_height=args.img_height,
                num_channels=1,
                num_classes=converter.vocab_size,
                rnn_hidden_size=args.hidden_size,
                dropout_rate=args.dropout_rate
            )
            logger.info("Using MCDropoutCRNN model")
        else:
            model = CRNN(
                img_height=args.img_height,
                num_channels=1,
                num_classes=converter.vocab_size,
                rnn_hidden_size=args.hidden_size
            )
            logger.info("Using standard CRNN model")
    elif args.model_type == 'parseq':
        model = PARSeq(
            vocab_size=90,  # Instead of len(vocab)
            max_label_length=35,  # Use 35 instead of args.max_length
            img_size=(args.img_height, args.img_width),
            embed_dim=384,
            encoder_num_layers=12,
            encoder_num_heads=6,
            decoder_num_layers=6,  # Use 6 decoder layers
            decoder_num_heads=8,
            decoder_mlp_ratio=4.0,
            dropout=args.dropout,
            sos_idx=1,
            eos_idx=2,
            pad_idx=0
        )
        logger.info("Using PARSeq model")
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Load model weights
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)  # Direct state dict
    model = model.to(device)
    logger.info(f"Loaded model weights from {args.model_path}")

    # Load test data
    test_transform = get_transforms('test', args.img_height, args.img_width)
    test_dataset = TeluguHWRDataset(
        data_file=os.path.join(args.data_root, args.test_file),
        root_dir=args.data_root,
        transform=test_transform,
        max_samples=args.max_samples
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    logger.info(f"Loaded test dataset with {len(test_dataset)} samples")

    # Evaluate model
    logger.info("Evaluating model...")
    model.eval()

    all_predictions = []
    all_targets = []

    # Character-level confusion matrix data
    char_predictions = []
    char_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            # Forward pass based on model type
            if args.model_type == 'crnn':
                logits = model(images)
                predictions = converter.decode(logits)
            elif args.model_type == 'parseq':
                from utils.tokenizer import PARSeqTokenizer

                parseq_tokenizer = PARSeqTokenizer(''.join(sorted(vocab)), max_length=args.max_length)

                outputs = model(images)

                # Use the PARSeq tokenizer to decode
                predictions = parseq_tokenizer.decode(outputs['logits'])

            # Store predictions and targets
            all_predictions.extend(predictions)
            all_targets.extend(labels)

            # Collect character-level data for confusion matrix
            for pred, target in zip(predictions, labels):
                # Get min length to avoid index errors
                min_len = min(len(pred), len(target))
                for i in range(min_len):
                    char_predictions.append(pred[i])
                    char_targets.append(target[i])

    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_targets)

    # Print results
    logger.info("\nEvaluation Results:")
    logger.info(f"Character Error Rate (CER): {metrics['character_error_rate']:.2f}%")
    logger.info(f"Word Error Rate (WER): {metrics['word_error_rate']:.2f}%")
    logger.info(f"Word Accuracy: {metrics['accuracy']:.2f}%")
    logger.info(f"Total Samples: {metrics['total_samples']}")

    logger.info("\nSample Predictions (GT vs Pred):")
    for gt, pred in zip(all_targets[:10], all_predictions[:10]):
        logger.info(f"GT: '{gt}' | Pred: '{pred}'")

    # Save results to JSON
    results = {
        "character_error_rate": metrics['character_error_rate'],
        "word_error_rate": metrics['word_error_rate'],
        "word_accuracy": metrics['accuracy'],
        "total_samples": metrics['total_samples'],
        "correct_words": metrics['correct_words'],
        "examples": [
            {"ground_truth": gt, "prediction": pred}
            for gt, pred in zip(all_targets[:20], all_predictions[:20])
        ]
    }

    with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # Generate visualizations

    # Character confusion matrix (limit to top most common characters)
    char_counts = {}
    for char in char_targets:
        char_counts[char] = char_counts.get(char, 0) + 1

    # Get top characters
    top_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:30]
    top_char_set = {char for char, _ in top_chars}

    # Filter chars for confusion matrix
    filtered_targets = []
    filtered_preds = []
    for gt, pred in zip(char_targets, char_predictions):
        if gt in top_char_set and pred in top_char_set:
            filtered_targets.append(gt)
            filtered_preds.append(pred)

    # Plot confusion matrix
    if filtered_targets:
        unique_chars = sorted(top_char_set)
        plot_confusion_matrix(
            filtered_targets, filtered_preds, unique_chars,
            os.path.join(args.output_dir, 'confusion_matrix.png')
        )

    # Plot error distribution
    plot_error_distribution(
        all_targets, all_predictions,
        os.path.join(args.output_dir, 'error_distribution.png')
    )

    logger.info(f"Results and visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Telugu Handwriting Recognition Model")

    # Data parameters
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory for dataset')
    parser.add_argument('--test_file', type=str, required=True,
                        help='Test annotation file')
    parser.add_argument('--vocab_file', type=str, required=True,
                        help='Path to vocabulary file')
    parser.add_argument('--output_dir', type=str, default='./evaluation',
                        help='Output directory for saving results')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum test samples to use (None for all)')

    # Model parameters
    parser.add_argument('--model_type', type=str, default='crnn', choices=['crnn', 'parseq'],
                        help='Type of model to evaluate')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--img_height', type=int, default=64,
                        help='Input image height')
    parser.add_argument('--img_width', type=int, default=256,
                        help='Input image width')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='LSTM hidden size (for CRNN)')
    parser.add_argument('--max_length', type=int, default=35,
                        help='Maximum sequence length (for PARSeq)')
    parser.add_argument('--num_permutations', type=int, default=6,
                        help='Number of permutations (for PARSeq)')
    parser.add_argument('--mc_dropout', action='store_true',
                        help='Use MC Dropout model')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout rate (for MC dropout)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (for PARSeq)')

    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA if available')

    args = parser.parse_args()
    main(args)
