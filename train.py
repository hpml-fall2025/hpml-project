"""
Training loop for FinBERT-CNN volatility forecasting
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import numpy as np


class Trainer:
    """Handles model training and validation"""

    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Loss function: MSE (as used in the paper)
        self.criterion = nn.MSELoss()

        # Optimizer: Adam with weight decay (L2 regularization)
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler (optional - can add later)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")

        # Setup checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0

        pbar = tqdm(
            self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs} [Train]"
        )
        for batch in pbar:
            # Move to device
            input_ids = batch["input_ids"].to(self.config.device)
            attention_mask = batch["attention_mask"].to(self.config.device)
            rv_true = batch["rv"].to(self.config.device).unsqueeze(1)  # (batch, 1)

            # Forward pass
            rv_pred = self.model(input_ids, attention_mask)
            loss = self.criterion(rv_pred, rv_true)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (optional but recommended)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Track loss
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = epoch_loss / len(self.train_loader)
        return avg_loss

    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            pbar = tqdm(
                self.val_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs} [Val]"
            )
            for batch in pbar:
                # Move to device
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                rv_true = batch["rv"].to(self.config.device).unsqueeze(1)

                # Forward pass
                rv_pred = self.model(input_ids, attention_mask)
                loss = self.criterion(rv_pred, rv_true)

                val_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

        avg_loss = val_loss / len(self.val_loader)
        return avg_loss

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "config": self.config,
        }

        # Regular checkpoint
        if (epoch + 1) % self.config.save_every == 0:
            path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save(checkpoint, path)
            print(f"Saved checkpoint: {path}")

        # Best model checkpoint
        if is_best:
            path = Path(self.config.checkpoint_dir) / "best_model.pt"
            torch.save(checkpoint, path)
            print(f"Saved best model: {path}")

    def train(self):
        """Full training loop"""
        print(f"\nStarting training for {self.config.num_epochs} epochs...")
        print(f"Device: {self.config.device}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print("-" * 60)

        for epoch in range(self.config.num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate(epoch)
            self.val_losses.append(val_loss)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Print summary
            print(
                f"Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            self.save_checkpoint(epoch, is_best)

        print("-" * 60)
        print(f"Training complete! Best validation loss: {self.best_val_loss:.6f}")

        return self.train_losses, self.val_losses


def load_checkpoint(model, checkpoint_path):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch'] + 1}")
    print(f"  Best val loss: {min(checkpoint['val_losses']):.6f}")
    return model
