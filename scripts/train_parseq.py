import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.telugu_dataset import TeluguDataset
from models.parseq import PARSeq
from trainers.parseq_trainer import PARSeqTrainer
from utils.tokenizer import Tokenizer
from utils.transforms import get_transforms

def parse_args():
    parser = argparse.ArgumentParser(description='Train PARSeq model for Telugu handwriting recognition')
    
    # Dataset parameters
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset root directory')
    parser.add_argument('--train_file', type=str, required=True, help='Train file name')
    parser.add_argument('--val_file', type=str, required=True, help='Validation file name')
    parser.add_argument('--vocab_file', type=str, required=True, help='Vocabulary file path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of training samples to use')
    parser.add_argument('--val_samples', type=int, default=None, help='Maximum number of validation samples to use')
    
    # Model parameters
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--enc_layers', type=int, default=6, help='Number of encoder layers')
    parser.add_argument('--dec_layers', type=int, default=6, help='Number of decoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--img_height', type=int, default=32, help='Image height')
    parser.add_argument('--img_width', type=int, default=128, help='Image width')
    parser.add_argument('--max_length', type=int, default=35, help='Maximum sequence length')
    parser.add_argument('--num_permutations', type=int, default=6, help='Number of permutations for training')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0007, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--patience', type=int, default=5, help='Patience for learning rate scheduler')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    
    args = parser.parse_args()
    return args

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def visualize_samples(dataset, output_path, num_samples=8):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(dataset))):
        image, text = dataset[i]
        axes[i].imshow(image.permute(1, 2, 0).numpy())
        axes[i].set_title(f"Text: {text}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    setup_logging()
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load datasets
    logging.info("Loading datasets...")
    tokenizer = Tokenizer(args.vocab_file)
    
    transform = get_transforms(args.img_height, args.img_width)
    
    train_dataset = TeluguDataset(
        os.path.join(args.data_root, args.train_file),
        root_dir=args.data_root,
        transform=transform,
        max_samples=args.max_samples
    )
    logging.info(f"Loaded {len(train_dataset)} samples")
    
    val_dataset = TeluguDataset(
        os.path.join(args.data_root, args.val_file),
        root_dir=args.data_root,
        transform=transform,
        max_samples=args.val_samples
    )
    logging.info(f"Loaded {len(val_dataset)} samples")
    
    # Visualize some samples
    visualize_samples(train_dataset, os.path.join(args.output_dir, 'train_samples.png'))
    print(f"Sample visualization saved to {os.path.join(args.output_dir, 'train_samples.png')}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda batch: (
            torch.stack([item[0] for item in batch]),
            [item[1] for item in batch]
        )
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda batch: (
            torch.stack([item[0] for item in batch]),
            [item[1] for item in batch]
        )
    )
    
    # Load vocabulary
    logging.info(f"Loading vocabulary from: {args.vocab_file}")
    with open(args.vocab_file, 'r', encoding='utf-8') as f:
        chars = [c.strip() for c in f.readlines()]
    logging.info(f"Loaded vocabulary with {len(chars)} characters")
    
    # Create configuration for trainer
    class Config:
        pass
    
    config = Config()
    config.device = device
    config.d_model = args.d_model
    config.num_heads = args.num_heads
    config.enc_layers = args.enc_layers
    config.dec_layers = args.dec_layers
    config.dim_feedforward = args.dim_feedforward
    config.dropout = args.dropout
    config.max_length = args.max_length
    config.num_classes = len(chars) + 3  # Add BOS, EOS, PAD tokens
    config.pad_id = 0
    config.bos_id = 1
    config.eos_id = 2
    config.img_height = args.img_height
    config.img_width = args.img_width
    config.learning_rate = args.learning_rate
    config.weight_decay = args.weight_decay
    config.patience = args.patience
    config.num_permutations = args.num_permutations
    config.output_dir = args.output_dir
    config.tokenizer = tokenizer
    
    # Create and train model
    logging.info("Creating PARSeq model...")
    trainer = PARSeqTrainer(config)
    
    logging.info("Starting training...")
    trainer.train(train_loader, val_loader, args.epochs)
    
    logging.info("Training completed!")

if __name__ == '__main__':
    main()
