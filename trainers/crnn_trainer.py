import time
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import tqdm
from utils.metrics import calculate_metrics
from trainers.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)

class CRNNTrainer(BaseTrainer):
    """
    Trainer for CRNN models.

    Args:
        model: CRNN model to train
        device: Device to train on (cuda/cpu)
        converter: CTCLabelConverter instance
        save_dir: Directory to save checkpoints
    """
    def __init__(self, model, device, converter, save_dir='checkpoints'):
        super().__init__(model, device, save_dir)
        self.converter = converter

    def train(self, train_loader, val_loader, epochs=30, lr=0.001, scheduler_type='plateau'):
        """
        Train the CRNN model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            lr: Initial learning rate
            scheduler_type: Type of learning rate scheduler ('plateau', 'cosine', etc.)

        Returns:
            Trained model and training history
        """
        # Setup model
        self.model = self.model.to(self.device)

        # Setup loss function and optimizer
        criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Setup scheduler
        if scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', factor=0.5, patience=3
            )
        elif scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=lr/100
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
            self.model.train()
            train_loss = 0.0
            batch_count = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for images, labels in progress_bar:
                batch_count += 1
                batch_size = images.size(0)
                images = images.to(self.device)

                # Forward pass
                log_probs = self.model(images)

                # Encode labels for CTC loss
                target_indices, target_lengths = self.converter.encode(labels)
                target_indices = target_indices.to(self.device)
                target_lengths = target_lengths.to(self.device)

                # Calculate input lengths (sequence length for each batch element)
                input_lengths = torch.full((batch_size,), log_probs.size(0), dtype=torch.long, device=self.device)

                # Calculate CTC loss
                loss = criterion(log_probs, target_indices, input_lengths, target_lengths)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()

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
            self.history['val_cer'].append(metrics['character_error_rate'])
            self.history['val_wer'].append(metrics['word_error_rate'])

            # Learning rate scheduler step
            if scheduler is not None:
                if scheduler_type == 'plateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

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

    def validate(self, data_loader, criterion):
        """
        Validate the model.

        Args:
            data_loader: Validation data loader
            criterion: Loss function

        Returns:
            Tuple of (validation loss, metrics dict)
        """
        self.model.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc="Validation"):
                batch_size = images.size(0)
                images = images.to(self.device)

                # Forward pass
                log_probs = self.model(images)

                # Encode targets for CTC loss
                target_indices, target_lengths = self.converter.encode(labels)
                target_indices = target_indices.to(self.device)
                target_lengths = target_lengths.to(self.device)

                # Calculate input lengths
                input_lengths = torch.full((batch_size,), log_probs.size(0), dtype=torch.long, device=self.device)

                # Calculate loss
                loss = criterion(log_probs, target_indices, input_lengths, target_lengths)
                val_loss += loss.item()

                # Decode predictions
                decoded_preds = self.converter.decode(log_probs)

                # Store for metrics calculation
                all_predictions.extend(decoded_preds)
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
