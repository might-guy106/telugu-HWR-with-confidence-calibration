import time
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import tqdm
from utils.metrics import calculate_metrics
from trainers.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)

class PARSeqTrainer(BaseTrainer):
    """
    Trainer for PARSeq models.

    Args:
        model: PARSeq model to train
        device: Device to train on (cuda/cpu)
        converter: CTCLabelConverter instance
        save_dir: Directory to save checkpoints
    """
    def __init__(self, model, device, converter, save_dir='checkpoints/parseq'):
        super().__init__(model, device, save_dir)
        self.converter = converter

    def convert_targets_to_indices(self, targets):
        """Convert string targets to indices for PARSeq."""
        batch_indices = []
        max_len = 0

        # First pass to find max length
        for text in targets:
            indices = []
            for char in text:
                idx = self.converter.char2idx.get(char, 0)  # Use 0 for unknown chars
                if idx > 0:  # Only add valid character indices
                    indices.append(idx)
            max_len = max(max_len, len(indices))
            batch_indices.append(indices)

        # Second pass to pad sequences
        padded_indices = []
        for indices in batch_indices:
            padded = indices + [self.converter.blank_idx] * (max_len - len(indices))
            padded_indices.append(padded)

        return torch.tensor(padded_indices, dtype=torch.long)

    def debug_converter(self):
        """Print debug information about the converter"""
        print("\nConverter Debug Info:")
        print(f"Blank token index: {self.converter.blank_idx}")
        print(f"First 10 char2idx mappings: {dict(list(self.converter.char2idx.items())[:10])}")
        print(f"First 10 idx2char mappings: {dict(list(self.converter.idx2char.items())[:10])}")

        # Basic test
        test_str = "test"
        indices, _ = self.converter.encode([test_str])
        print(f"Test string '{test_str}' encodes to: {indices.tolist()}")

        # Call this function before training
        # self.debug_converter()


    def train(self, train_loader, val_loader, epochs=50, lr=0.0007):
        """
        Train the PARSeq model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            lr: Learning rate

        Returns:
            Trained model and training history
        """
        # Setup model
        self.model = self.model.to(self.device)

        # Set up optimizer and loss
        from models.parseq import PermutationLoss  # Import here to avoid circular imports

        criterion = PermutationLoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=0.01,
            betas=(0.9, 0.99)
        )

        # Learning rate scheduler - OneCycleLR
        total_steps = len(train_loader) * epochs
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=10.0,
            final_div_factor=1000.0
        )

        # Initialize history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'val_cer': [],
            'val_wer': [],
            'learning_rates': []
        }

        # Best model tracking
        best_val_loss = float('inf')
        best_val_cer = float('inf')

        # Start timer
        self.start_time = time.time()

        # Training loop
        logger.info(f"Starting training for {epochs} epochs...")
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            batch_count = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for images, labels in progress_bar:
                batch_count += 1

                # Move data to device
                images = images.to(self.device)

                # Convert string targets to indices
                target_indices = self.convert_targets_to_indices(labels).to(self.device)

                # Forward pass
                outputs = self.model(images, target_indices, train=True)

                # Calculate loss
                loss = criterion(
                    outputs["logits"],
                    target_indices,
                    outputs["permutations"]
                )

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # Accumulate loss
                train_loss += loss.item()

                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})

            # Calculate average training loss
            train_loss /= len(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['learning_rates'].append(optimizer.param_groups[0]['lr'])

            # Validation phase
            val_loss, metrics = self.validate(val_loader, criterion)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(metrics['accuracy'])
            self.history['val_cer'].append(metrics['character_error_rate'])
            self.history['val_wer'].append(metrics['word_error_rate'])

            # Print epoch summary
            logger.info(f"Epoch {epoch+1}/{epochs} Summary - "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, "
                       # f"Val Acc: {metrics['accuracy']:.2f}%, "
                       f"Val CER: {metrics['character_error_rate']:.2f}%, "
                       f"Val WER: {metrics['word_error_rate']:.2f}%")

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

            # Save best model based on validation CER
            if metrics['character_error_rate'] < best_val_cer:
                best_val_cer = metrics['character_error_rate']
                self.save_checkpoint(
                    epoch=epoch,
                    optimizer=optimizer,
                    metrics=metrics,
                    filename="best_cer_model.pth"
                )
                logger.info(f"Saved best CER model with val_cer: {metrics['character_error_rate']:.2f}%")

            # Save regular checkpoint
            if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
                self.save_checkpoint(
                    epoch=epoch,
                    optimizer=optimizer,
                    metrics=metrics,
                    filename=f"checkpoint_epoch_{epoch+1}.pth"
                )

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

    def validate(self, data_loader, criterion, quick_mode=True):
        """
        Validate the model.

        Args:
            data_loader: Validation data loader
            criterion: Loss function
            quick_mode: Whether to skip the slow autoregressive decoding for development

        Returns:
            Tuple of (validation loss, metrics dict)
        """
        self.model.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc="Validation"):
                # Move data to device
                images = images.to(self.device)

                # Convert string targets to indices
                target_indices = self.convert_targets_to_indices(labels).to(self.device)

                # Forward pass with teacher forcing (for loss)
                outputs = self.model(images, target_indices, train=True)

                # Calculate loss
                loss = criterion(
                    outputs["logits"],
                    target_indices,
                    outputs["permutations"]
                )
                val_loss += loss.item()

                # Forward pass with autoregressive decoding (for accuracy metrics)
                if not quick_mode:
                    # Full autoregressive decoding (slow but accurate)
                    decode_outputs = self.model(images, train=False)
                    predictions = decode_outputs["predictions"]

                    # Debug: Print shapes and a sample of raw predictions
                    if batch_idx == 0 and len(predictions) > 0:
                        print(f"\nDebug - Prediction shape: {predictions.shape}")
                        print(f"Debug - First prediction: {predictions[0].tolist()[:10]} (first 10 indices only)")

                    # Convert indices to text
                    pred_texts = []
                    for pred in predictions:
                        # Skip pad/blank tokens (0) and convert valid tokens
                        text = ''.join(self.converter.idx2char.get(idx.item(), '')
                                        for idx in pred if idx.item() > 0)  # Only use indices > 0
                        pred_texts.append(text)
                        print(f"Sample prediction: '{text}'")  # Show a prediction
                else:
                    # Quick mode: Skip decoding and just use empty strings during development
                    pred_texts = [""] * len(labels)

                # Store for metrics calculation
                all_predictions.extend(pred_texts)
                all_targets.extend(labels)

        # Calculate average loss
        val_loss /= len(data_loader)

        # Calculate metrics
        metrics = calculate_metrics(all_predictions, all_targets)

        # Log a few examples
        logger.info("Validation Examples (GT vs Pred):")
        for gt, pred in zip(all_targets[:3], all_predictions[:3]):
            logger.info(f"GT: '{gt}' | Pred: '{pred}'")

        return val_loss, metrics
