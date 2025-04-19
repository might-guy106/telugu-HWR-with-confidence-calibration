import torch
import os
import logging
from torch.utils.data import DataLoader

from data.dataset import TeluguHWRDataset
from data.transforms import get_transforms
from models.crnn import CRNN
from utils.ctc_decoder import CTCLabelConverter
from trainers.crnn_trainer import CRNNTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    """Test the restructured code."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Settings
    data_root = "/path/to/your/data"  # Update this path
    img_height = 64
    img_width = 256
    batch_size = 32
    vocab_file = "path/to/vocabulary.txt"  # Update this path

    # Create transforms
    train_transform = get_transforms('train', img_height, img_width)
    val_transform = get_transforms('val', img_height, img_width)

    # Load a small subset of data for testing
    train_dataset = TeluguHWRDataset(
        data_file=os.path.join(data_root, "train.txt"),
        root_dir=data_root,
        transform=train_transform,
        max_samples=100  # Small batch for testing
    )

    val_dataset = TeluguHWRDataset(
        data_file=os.path.join(data_root, "val.txt"),
        root_dir=data_root,
        transform=val_transform,
        max_samples=50  # Small batch for testing
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    # Load vocabulary
    vocab = set()
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
            char = line.strip()
            if char:  # Skip empty lines
                vocab.add(char)

    # Create converter
    converter = CTCLabelConverter(vocab)

    # Create model
    model = CRNN(
        img_height=img_height,
        num_channels=1,
        num_classes=converter.vocab_size,
        rnn_hidden_size=256
    )

    # Create trainer
    trainer = CRNNTrainer(
        model=model,
        device=device,
        converter=converter,
        save_dir='./test_checkpoints'
    )

    # Run a tiny training job to test
    logger.info("Starting a test training run...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,  # Just a couple of epochs for testing
        lr=0.001
    )

    logger.info("Test completed successfully!")

if __name__ == "__main__":
    main()
