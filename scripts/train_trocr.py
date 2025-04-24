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
from data.transforms import get_trocr_transforms
from models.trocr import TrOCRModel
from trainers.trocr_trainer import TrOCRTrainer
from utils.visualization import plot_training_history

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('logs/trocr_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    logger.info(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create data transforms for TrOCR
    train_transform = get_trocr_transforms('train', args.img_height, args.img_width)
    val_transform = get_trocr_transforms('val', args.img_height, args.img_width)

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

    # Create TrOCR model
    logger.info("Creating TrOCR model...")
    model = TrOCRModel(
        pretrained_model_name=args.pretrained_model,
        max_length=args.max_length,
        is_fine_tuned=args.fine_tuned,
        device=device
    )

    # Create trainer
    trainer = TrOCRTrainer(
        model=model,
        device=device,
        save_dir=args.output_dir
    )

    # Load checkpoint if provided
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        trainer.load_checkpoint(args.resume, optimizer)

    # Train model
    logger.info("Starting training...")
    trained_model, history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.learning_rate
    )

    # Plot training history
    plot_training_history(
        history,
        os.path.join(args.output_dir, 'trocr_training_history.png')
    )

    logger.info("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TrOCR for Telugu Handwriting Recognition")

    # Data parameters
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory for dataset')
    parser.add_argument('--train_file', type=str, default='train.txt',
                        help='Training annotation file')
    parser.add_argument('--val_file', type=str, default='val.txt',
                        help='Validation annotation file')
    parser.add_argument('--output_dir', type=str, default='./output/trocr',
                        help='Output directory for saving models and results')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum training samples to use (None for all)')
    parser.add_argument('--val_samples', type=int, default=None,
                        help='Maximum validation samples to use (None for all)')

    # Model parameters
    parser.add_argument('--pretrained_model', type=str,
                        default='microsoft/trocr-base-handwritten',
                        help='Pretrained model name or path')
    parser.add_argument('--fine_tuned', action='store_true',
                        help='Use fine-tuned model for Telugu')
    parser.add_argument('--img_height', type=int, default=384,
                        help='Input image height')
    parser.add_argument('--img_width', type=int, default=384,
                        help='Input image width')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Initial learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA if available')

    args = parser.parse_args()
    main(args)
