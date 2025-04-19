import os
import time
import torch
from datetime import datetime
import logging
from pathlib import Path
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class BaseTrainer:
    """
    Base class for model trainers.

    This provides common functionality for all model trainers.

    Args:
        model: Model to train
        device: Device to train on
        save_dir: Directory to save checkpoints
    """
    def __init__(self, model, device, save_dir='checkpoints'):
        """Initialize the trainer."""
        self.model = model
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.history = {}
        self.start_time = None
        self.best_metrics = {}

    def save_checkpoint(self, epoch, optimizer, metrics, filename):
        """
        Save model checkpoint to disk.

        Args:
            epoch: Current epoch
            optimizer: Optimizer state
            metrics: Dictionary of metrics
            filename: Filename to save checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }

        torch.save(checkpoint, str(self.save_dir / filename))
        logger.info(f"Saved checkpoint to {str(self.save_dir / filename)}")

    def load_checkpoint(self, checkpoint_path, optimizer=None):
        """
        Load model checkpoint from disk.

        Args:
            checkpoint_path: Path to checkpoint file
            optimizer: Optional optimizer to load state

        Returns:
            Dictionary with checkpoint info
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']})")
        return checkpoint

    def update_history(self, metrics):
        """
        Update training history with new metrics.

        Args:
            metrics: Dictionary of metrics to update
        """
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

    def plot_history(self, output_file=None):
        """
        Plot training history metrics.

        Args:
            output_file: Optional path to save the plot
        """
        if not self.history:
            logger.warning("No history to plot")
            return

        n_metrics = len(self.history)
        fig_height = 4 * ((n_metrics + 1) // 2)  # Adjust height based on number of metrics

        plt.figure(figsize=(12, fig_height))

        for i, (key, values) in enumerate(self.history.items()):
            plt.subplot((n_metrics + 1) // 2, 2, i + 1)
            plt.plot(values)
            plt.title(key)
            plt.xlabel('Epoch')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file)
            logger.info(f"Saved history plot to {output_file}")
        else:
            plt.show()

    def log_elapsed_time(self):
        """Log elapsed training time."""
        if self.start_time:
            elapsed = time.time() - self.start_time
            if elapsed < 60:
                logger.info(f"Time elapsed: {elapsed:.1f} seconds")
            elif elapsed < 3600:
                logger.info(f"Time elapsed: {elapsed/60:.1f} minutes")
            else:
                logger.info(f"Time elapsed: {elapsed/3600:.1f} hours")
