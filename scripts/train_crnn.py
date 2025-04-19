import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import argparse
import torch
import logging
from torch.utils.data import DataLoader

from data.dataset import TeluguHWRDataset
from data.transforms import get_transforms
from data.analysis import analyze_dataset, visualize_samples
from models.crnn import CRNN
from models.mc_dropout_crnn import MCDropoutCRNN
from utils.ctc_decoder import CTCLabelConverter
from trainers.crnn_trainer import CRNNTrainer
from utils.visualization import plot_training_history

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    logger.info(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create data transforms
    train_transform = get_transforms('train', args.img_height, args.img_width)
    val_transform = get_transforms('val', args.img_height, args.img_width)

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = TeluguHWRDataset(
        data_file=os.path.join(args.data_root, args.train_file),
        root_dir=args.data_root,
        transform=train_transform,
        max_samples=args.max_samples
    )

    val_dataset = TeluguHWRDataset(
        data_file=os.path.join(args.data_root, args.val_file),
        root_dir=args.data_root,
        transform=val_transform,
        max_samples=args.val_samples
    )

    # Visualize some samples
    visualize_samples(
        train_dataset,
        num_samples=10,
        cols=5,
        save_path=os.path.join(args.output_dir, 'train_samples.png')
    )

    # Handle vocabulary
    if args.vocab_file:
        # Load vocabulary from file
        logger.info(f"Loading vocabulary from: {args.vocab_file}")
        vocab = set()
        with open(args.vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                char = line.strip()
                if char:  # Skip empty lines
                    vocab.add(char)
        logger.info(f"Loaded vocabulary with {len(vocab)} characters")
    else:
        # Analyze dataset and get vocabulary
        logger.info("Analyzing dataset and creating vocabulary...")
        train_vocab = analyze_dataset(train_dataset, save_dir=args.output_dir)
        val_vocab = analyze_dataset(val_dataset, save_dir=args.output_dir)
        # Combine vocabularies
        vocab = train_vocab.union(val_vocab)
        logger.info(f"Created vocabulary with {len(vocab)} characters")

        # Save vocabulary to file
        vocab_output_path = os.path.join(args.output_dir, 'vocabulary.txt')
        logger.info(f"Saving vocabulary to: {vocab_output_path}")
        with open(vocab_output_path, 'w', encoding='utf-8') as f:
            for char in sorted(vocab):
                f.write(char + '\n')

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Create converter
    converter = CTCLabelConverter(vocab)

    # Create model
    logger.info("Creating model...")
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

    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create trainer
    trainer = CRNNTrainer(
        model=model,
        device=device,
        converter=converter,
        save_dir=args.output_dir
    )

    # Load checkpoint if provided
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        trainer.load_checkpoint(args.resume, optimizer)

    # Train model
    logger.info("Starting training...")
    trained_model, history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.learning_rate,
        scheduler_type=args.scheduler
    )

    # Plot training history
    plot_training_history(
        history,
        os.path.join(args.output_dir, 'training_history.png')
    )

    logger.info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CRNN for Telugu Handwriting Recognition")

    # Data parameters
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory for dataset')
    parser.add_argument('--train_file', type=str, default='train.txt',
                        help='Training annotation file')
    parser.add_argument('--val_file', type=str, default='val.txt',
                        help='Validation annotation file')
    parser.add_argument('--vocab_file', type=str, default=None,
                        help='Path to existing vocabulary file')
    parser.add_argument('--output_dir', type=str, default='./output/crnn',
                        help='Output directory for saving models and results')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum training samples to use (None for all)')
    parser.add_argument('--val_samples', type=int, default=None,
                        help='Maximum validation samples to use (None for all)')

    # Model parameters
    parser.add_argument('--img_height', type=int, default=64,
                        help='Input image height')
    parser.add_argument('--img_width', type=int, default=256,
                        help='Input image width')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='LSTM hidden size')
    parser.add_argument('--mc_dropout', action='store_true',
                        help='Use MC Dropout model for uncertainty estimation')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout rate (for MC dropout model)')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['plateau', 'cosine'],
                        help='Learning rate scheduler type')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA if available')

    args = parser.parse_args()
    main(args)
