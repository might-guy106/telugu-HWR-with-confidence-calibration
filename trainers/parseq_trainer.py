import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from models.parseq import PARSeq
from utils.metrics import compute_accuracy
import logging

class PARSeqTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # Create model
        self.model = PARSeq(
            d_model=config.d_model,
            num_heads=config.num_heads,
            enc_layers=config.enc_layers,
            dec_layers=config.dec_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            max_length=config.max_length,
            num_classes=config.num_classes,
            pad_id=config.pad_id,
            eos_id=config.eos_id,
            image_height=config.img_height,
            image_width=config.img_width
        ).to(self.device)
        
        logging.info(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.learning_rate,  # Use directly without scaling
            weight_decay=config.weight_decay
        )
        
        # Create scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=config.patience, factor=0.5
        )
        
        # Create criterion
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.pad_id)
        
        # Create scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()
        
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0

    def train(self, train_loader, val_loader, epochs):
        logging.info(f"Starting training with initial lr={self.optimizer.param_groups[0]['lr']}")
        start_time = time.time()
        
        for epoch in range(1, epochs+1):
            train_loss, train_accuracy = self._train_epoch(train_loader, epoch, epochs)
            val_loss, val_accuracy = self._validate(val_loader, epoch, epochs)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save checkpoint if improved
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_accuracy = val_accuracy
                self.save_checkpoint(os.path.join(self.config.output_dir, 'best_model.pth'))
            
        # Save final model
        self.save_checkpoint(os.path.join(self.config.output_dir, 'final_model.pth'))
        
        elapsed = time.time() - start_time
        logging.info(f"Time elapsed: {elapsed:.1f} seconds")
        
    def _train_epoch(self, train_loader, epoch, total_epochs):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        nan_batches = 0
        
        from tqdm import tqdm
        progress_bar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch}/{total_epochs} [Train]",
            leave=True
        )
        
        for i, batch in enumerate(progress_bar):
            images, texts = batch
            
            # Debug text encoding
            if i < 2:  # Only print for first two batches
                for j in range(2):  # Print first two examples
                    if j < len(texts):
                        print(f"Encoding text: '{texts[j]}'")
                        indices = self.config.tokenizer.encode(texts[j])
                        print(f"Encoded indices: {indices}")
            
            images = images.to(self.device)
            encoded = [self.config.tokenizer.encode(text) for text in texts]
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.amp.autocast(device_type='cuda', enabled=True):  # Updated autocast API
                outputs = self.model(images, encoded, self.config.num_permutations)
                loss = self.model.compute_loss(outputs, encoded, self.criterion)
            
            # Skip batch with NaN loss
            if torch.isnan(loss).item():
                nan_batches += 1
                progress_bar.set_postfix(loss="NaN", NaN_batches=nan_batches)
                continue
                
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Add gradient clipping to prevent exploding gradients
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Calculate accuracy
            batch_acc = self.model.calculate_accuracy(outputs, encoded)
            
            running_loss += loss.item()
            correct += batch_acc[0]
            total += batch_acc[1]
            
            # Update progress bar
            progress_bar.set_postfix(
                loss=f"{running_loss/(i+1-nan_batches):.4f}",
                acc=f"{correct/total*100:.2f}%",
                NaN_batches=nan_batches
            )
        
        epoch_loss = running_loss / (len(train_loader) - nan_batches) if len(train_loader) > nan_batches else float('inf')
        epoch_accuracy = correct / total if total > 0 else 0
        
        if nan_batches > 0:
            logging.warning(f"Skipped {nan_batches}/{len(train_loader)} batches with NaN loss")
        
        return epoch_loss, epoch_accuracy
        
    def _validate(self, val_loader, epoch, total_epochs):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            from tqdm import tqdm
            progress_bar = tqdm(
                val_loader, 
                desc=f"Epoch {epoch}/{total_epochs} [Val]",
                leave=True
            )
            
            for i, batch in enumerate(progress_bar):
                images, texts = batch
                images = images.to(self.device)
                encoded = [self.config.tokenizer.encode(text) for text in texts]
                
                # Forward pass
                outputs = self.model(images, encoded, 1)  # No permutations during validation
                loss = self.model.compute_loss(outputs, encoded, self.criterion)
                
                # Calculate accuracy
                batch_acc = self.model.calculate_accuracy(outputs, encoded)
                
                running_loss += loss.item()
                correct += batch_acc[0]
                total += batch_acc[1]
                
                # Update progress bar
                progress_bar.set_postfix(
                    loss=f"{running_loss/(i+1):.4f}",
                    acc=f"{correct/total*100:.2f}%"
                )
        
        epoch_loss = running_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        epoch_accuracy = correct / total if total > 0 else 0
        
        return epoch_loss, epoch_accuracy
        
    def save_checkpoint(self, checkpoint_path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
        }
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Saved checkpoint to {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            logging.warning(f"Checkpoint file {checkpoint_path} not found")
            return
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_accuracy = checkpoint['best_val_accuracy']
        logging.info(f"Loaded checkpoint from {checkpoint_path}")
