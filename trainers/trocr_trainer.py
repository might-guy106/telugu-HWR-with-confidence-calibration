import os
import time
import torch
import logging
import numpy as np
from tqdm import tqdm
from trainers.base_trainer import BaseTrainer
from transformers import get_linear_schedule_with_warmup
from utils.metrics import calculate_metrics

logger = logging.getLogger(__name__)

class TrOCRTrainer(BaseTrainer):
    """
    Trainer for TrOCR models.

    Args:
        model: TrOCR model to train
        device: Device to use for training
        save_dir: Directory to save checkpoints
    """

    def __init__(self, model, device, save_dir='checkpoints'):
        super().__init__(model, device, save_dir)

    def train(self, train_loader, val_loader, epochs=10, lr=5e-5, scheduler_type='linear_warmup'):
        """
        Train the TrOCR model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            lr: Learning rate
            scheduler_type: Type of learning rate scheduler

        Returns:
            Tuple of (trained model, training history)
        """
        # Set model to train mode
        self.model.train()

        # Setup optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=0.01
        )

        # Setup scheduler
        if scheduler_type == 'linear_warmup':
            # Calculate total steps
            total_steps = len(train_loader) * epochs
            warmup_steps = total_steps // 10  # 10% warmup

            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:
            scheduler = None

        # Initialize history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_cer': [],
            'val_wer': [],
            'learning_rates': []
        }

        # Best model tracking
        best_val_loss = float('inf')
        best_cer = float('inf')

        # Start timer
        self.start_time = time.time()

        # Training loop
        logger.info(f"Starting training for {epochs} epochs...")
        for epoch in range(epochs):
            # Training phase
            train_loss = 0.0
            batch_count = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for images, labels in progress_bar:
                batch_count += 1

                # Move images to device
                if isinstance(images, torch.Tensor):
                    images = images.to(self.device)

                # Forward pass with labels
                outputs = self.model(images, labels)

                # Get loss
                loss = outputs.loss

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                if scheduler:
                    scheduler.step()

                # Accumulate loss
                train_loss += loss.item()

                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})

            # Calculate average training loss
            train_loss /= batch_count
            self.history['train_loss'].append(train_loss)
            self.history['learning_rates'].append(optimizer.param_groups[0]['lr'])

            # Validation phase
            val_loss, metrics = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_cer'].append(metrics['character_error_rate'])
            self.history['val_wer'].append(metrics['word_error_rate'])

            # Print epoch summary
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch+1}/{epochs} Summary - "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, "
                       f"Val CER: {metrics['character_error_rate']:.2f}%, "
                       f"Val WER: {metrics['word_error_rate']:.2f}%, "
                       f"LR: {current_lr:.6f}")

            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(
                    epoch=epoch,
                    optimizer=optimizer,
                    metrics=metrics,
                    filename="best_loss_model.pth"
                )
                logger.info(f"Saved best loss model with val_loss: {val_loss:.4f}")

            # Save best model based on CER
            if metrics['character_error_rate'] < best_cer:
                best_cer = metrics['character_error_rate']
                self.save_checkpoint(
                    epoch=epoch,
                    optimizer=optimizer,
                    metrics=metrics,
                    filename="best_cer_model.pth"
                )
                logger.info(f"Saved best CER model with val_cer: {metrics['character_error_rate']:.2f}%")

        # Save final model
        self.save_checkpoint(
            epoch=epochs-1,
            optimizer=optimizer,
            metrics=metrics,
            filename="final_model.pth"
        )

        # Log total training time
        self.log_elapsed_time()

        return self.model, self.history

    def validate(self, data_loader):
        """
        Validate the model on a validation set.

        Args:
            data_loader: Validation data loader

        Returns:
            Tuple of (validation loss, metrics dict)
        """
        self.model.eval()
        val_loss = 0.0
        batch_count = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc="Validation"):
                batch_count += 1

                # Move images to device
                if isinstance(images, torch.Tensor):
                    images = images.to(self.device)

                # Forward pass with labels
                outputs = self.model(images, labels)

                # Accumulate loss
                val_loss += outputs.loss.item()

                # Generate predictions
                generate_outputs = self.model.generate(images)
                predictions = generate_outputs['predicted_text']

                # Store for metrics calculation
                all_predictions.extend(predictions)
                all_targets.extend(labels)

        # Calculate average loss
        val_loss /= batch_count

        # Calculate metrics
        metrics = calculate_metrics(all_predictions, all_targets)

        # Log a few examples
        logger.info("Validation Examples (GT vs Pred):")
        for gt, pred in zip(all_targets[:3], all_predictions[:3]):
            logger.info(f"GT: '{gt}' | Pred: '{pred}'")

        return val_loss, metrics

    def save_checkpoint(self, epoch, optimizer, metrics, filename):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            optimizer: Optimizer state
            metrics: Evaluation metrics
            filename: Filename to save to
        """
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

        # For transformers models, save both the traditional checkpoint and the HF format
        # Traditional checkpoint for compatibility with your framework
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }

        torch.save(checkpoint, os.path.join(self.save_dir, filename))

        # Save the model in HF format (without .pth extension)
        model_dir = os.path.join(self.save_dir, filename.replace('.pth', ''))
        self.model.save_pretrained(model_dir)

        logger.info(f"Saved checkpoint to {os.path.join(self.save_dir, filename)}")
        logger.info(f"Saved model in HF format to {model_dir}")

    def load_checkpoint(self, checkpoint_path, optimizer=None):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
            optimizer: Optional optimizer to load state

        Returns:
            Dictionary with checkpoint info
        """
        # Check if the path is a directory (HF format) or a file (traditional checkpoint)
        if os.path.isdir(checkpoint_path):
            # Load from HF format
            self.model = self.model.__class__.from_pretrained(checkpoint_path, device=self.device)

            # Try to load additional info from a metadata file if exists
            metadata_path = os.path.join(checkpoint_path, 'metadata.pt')
            if os.path.isfile(metadata_path):
                metadata = torch.load(metadata_path, map_location=self.device)
                logger.info(f"Loaded metadata from {metadata_path}")
                return metadata
            else:
                # Return a default checkpoint dict
                return {'epoch': 0, 'metrics': {}}
        else:
            # Load traditional checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.model.load_state_dict(checkpoint['model_state_dict'])

            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return checkpoint
