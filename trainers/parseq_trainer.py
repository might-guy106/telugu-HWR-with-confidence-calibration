import os
import time
import torch
import logging
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from models.parseq import PARSeqLoss  # This will now work with the added class
from utils.metrics import calculate_metrics
from trainers.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)

class PARSeqTrainer(BaseTrainer):
    """
    Trainer for PARSeq models.

    Args:
        model: PARSeq model to train
        device: Device to train on (cuda/cpu)
        tokenizer: Tokenizer for encoding/decoding text
        save_dir: Directory to save checkpoints
    """

    def __init__(self, model, device, tokenizer, save_dir='checkpoints/parseq'):
        super().__init__(model, device, save_dir)
        self.tokenizer = tokenizer

        # Create loss function with label smoothing
        self.criterion = PARSeqLoss(ignore_index=tokenizer.pad_id, label_smoothing=0.1)

    def train(self, train_loader, val_loader, epochs=30, lr=0.0007, weight_decay=0.0001):
        """Train the PARSeq model."""
        self.model = self.model.to(self.device)

        # Start with a lower learning rate for stability
        initial_lr = lr * 0.01

        # Create optimizer with conservative settings
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=initial_lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # LR scheduler - OneCycleLR works well with PARSeq
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,  # Warm up for 10% of training
            div_factor=25,  # initial_lr = max_lr/25
            final_div_factor=1000,  # min_lr = initial_lr/1000
        )

        # Initialize history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_cer': [],
            'val_wer': [],
            'val_acc': [],
            'learning_rates': []
        }

        # Initialize best metrics
        best_val_loss = float('inf')
        best_val_cer = float('inf')
        self.start_time = time.time()

        # Training loop
        logger.info(f"Starting training with initial lr={initial_lr:.6f}")

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            valid_batches = 0
            nan_batches = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for images, labels in progress_bar:
                # Move data to device
                images = images.to(self.device)

                # Tokenize labels
                encoded_targets, _ = self.tokenizer.encode(labels)
                targets = torch.tensor(encoded_targets, device=self.device)

                # Zero gradients
                optimizer.zero_grad()

                try:
                    # Forward pass
                    outputs = self.model(images, targets)
                    logits = outputs["logits"]
                    tgt_output = outputs["targets"]

                    # Check for NaN in logits and skip if found
                    if torch.isnan(logits).any():
                        nan_batches += 1
                        progress_bar.set_postfix({'loss': 'NaN', 'NaN batches': nan_batches})
                        continue

                    # Calculate loss
                    loss = self.criterion(logits, tgt_output)

                    # Check for NaN loss
                    if torch.isnan(loss).item() or torch.isinf(loss).item():
                        nan_batches += 1
                        progress_bar.set_postfix({'loss': 'NaN', 'NaN batches': nan_batches})
                        continue

                    # Backward and optimize
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()

                    # Accumulate stats
                    train_loss += loss.item()
                    valid_batches += 1

                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': loss.item(),
                        'lr': scheduler.get_last_lr()[0],
                        'NaN batches': nan_batches
                    })

                except Exception as e:
                    logger.error(f"Error in batch: {str(e)}")
                    nan_batches += 1
                    continue

            # Log NaN batches
            if nan_batches > 0:
                logger.warning(f"Skipped {nan_batches}/{len(train_loader)} batches")

            # Calculate average loss
            if valid_batches > 0:
                train_loss /= valid_batches
            else:
                train_loss = float('nan')

            self.history['train_loss'].append(train_loss)
            self.history['learning_rates'].append(scheduler.get_last_lr()[0])

            # Validation
            if valid_batches > 0:
                val_loss, metrics = self.validate(val_loader)

                self.history['val_loss'].append(val_loss)
                self.history['val_cer'].append(metrics['character_error_rate'])
                self.history['val_wer'].append(metrics['word_error_rate'])
                self.history['val_acc'].append(metrics['accuracy'])

                # Log summary
                logger.info(f"Epoch {epoch+1}/{epochs} Summary - "
                          f"Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, "
                          f"Val CER: {metrics['character_error_rate']:.2f}%, "
                          f"Val WER: {metrics['word_error_rate']:.2f}%, "
                          f"Val Acc: {metrics['accuracy']:.2f}%")

                # Save best models
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(
                        epoch=epoch,
                        optimizer=optimizer,
                        metrics=metrics,
                        filename="best_loss_model.pth"
                    )

                if metrics['character_error_rate'] < best_val_cer:
                    best_val_cer = metrics['character_error_rate']
                    self.save_checkpoint(
                        epoch=epoch,
                        optimizer=optimizer,
                        metrics=metrics,
                        filename="best_cer_model.pth"
                    )

                # Save checkpoint every few epochs
                if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                    self.save_checkpoint(
                        epoch=epoch,
                        optimizer=optimizer,
                        metrics=metrics,
                        filename=f"checkpoint_epoch_{epoch+1}.pth"
                    )

        # Final save
        self.save_checkpoint(
            epoch=epochs-1,
            optimizer=optimizer,
            metrics=metrics if valid_batches > 0 else {"character_error_rate": 100.0},
            filename="final_model.pth"
        )

        self.log_elapsed_time()
        return self.model, self.history

    def validate(self, data_loader):
        """Validate the model with improved debugging."""
        self.model.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []
        valid_batches = 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(data_loader, desc="Validation")):
                try:
                    # Move data to device
                    images = images.to(self.device)

                    # Debug info for the first batch
                    if batch_idx == 0:
                        print(f"\nDEBUG: Validating first batch with {len(labels)} samples")
                        print(f"DEBUG: First few labels: {labels[:3]}")

                    # Get model predictions (inference mode)
                    outputs = self.model(images)
                    pred_ids = outputs["logits"]

                    # Debug token sequences
                    if batch_idx == 0:
                        print(f"DEBUG: Raw prediction shape: {pred_ids.shape}")
                        print(f"DEBUG: First prediction tokens: {pred_ids[0]}")

                    # We need to filter out SOS, PAD, and EOS tokens before decoding
                    # This is often the source of empty predictions
                    pred_texts = []
                    for pred in pred_ids:
                        # Convert to list and process
                        tokens = pred.cpu().tolist()

                        # Remove SOS token (always first)
                        if tokens and tokens[0] == self.tokenizer.sos_id:
                            tokens = tokens[1:]

                        # Find EOS token and truncate
                        if self.tokenizer.eos_id in tokens:
                            eos_idx = tokens.index(self.tokenizer.eos_id)
                            tokens = tokens[:eos_idx]

                        # Filter PAD tokens
                        tokens = [t for t in tokens if t != self.tokenizer.pad_id]

                        # Convert indices to characters
                        text = ''.join([self.tokenizer.idx_to_char.get(idx, '') for idx in tokens])
                        pred_texts.append(text)

                    # Debug decoded texts
                    if batch_idx == 0:
                        print(f"DEBUG: First few decoded texts: {pred_texts[:3]}")

                    # Store for metrics calculation
                    all_predictions.extend(pred_texts)
                    all_targets.extend(labels)

                    # For validation loss, use teacher forcing mode
                    try:
                        encoded_targets, _ = self.tokenizer.encode(labels)
                        targets = torch.tensor(encoded_targets, device=self.device)

                        # Forward pass with targets for loss calculation
                        outputs_with_targets = self.model(images, targets)
                        logits = outputs_with_targets["logits"]
                        tgt_output = outputs_with_targets["targets"]

                        # Calculate loss if no NaN
                        if not torch.isnan(logits).any():
                            loss = self.criterion(logits, tgt_output)
                            if not torch.isnan(loss).item() and not torch.isinf(loss).item():
                                val_loss += loss.item()
                                valid_batches += 1
                    except Exception as e:
                        print(f"DEBUG: Error in validation loss calculation: {str(e)}")

                except Exception as e:
                    print(f"DEBUG: Error in validation batch {batch_idx}: {str(e)}")
                    continue

        # Calculate average loss
        if valid_batches > 0:
            val_loss /= valid_batches
        else:
            val_loss = float('nan')

        # Calculate metrics
        if len(all_predictions) > 0:
            metrics = calculate_metrics(all_predictions, all_targets)
        else:
            metrics = {
                'character_error_rate': 100.0,
                'word_error_rate': 100.0,
                'accuracy': 0.0,
                'total_samples': len(data_loader.dataset),
                'correct_words': 0
            }

        # Log examples
        print("\nValidation Examples (GT vs Pred):")
        for gt, pred in zip(all_targets[:5], all_predictions[:5]):
            print(f"GT: '{gt}' | Pred: '{pred}'")

        return val_loss, metrics
