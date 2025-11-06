"""
Training script for FinBERT sentiment analysis model.
Includes mixed precision training, gradient accumulation, validation, and checkpointing.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AdamW
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
import config
from src.finbert_model import create_finbert_model
from src.data_loader import load_finbert_data, create_dataloaders


class FinBERTTrainer:
    """
    Trainer class for FinBERT model with full training loop.
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device=None,
        use_amp=None
    ):
        """
        Args:
            model: FinBERT model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            device: Device to train on (default: config.DEVICE)
            use_amp: Use automatic mixed precision (default: config.USE_AMP)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or config.DEVICE
        self.use_amp = use_amp if use_amp is not None else config.USE_AMP
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Calculate total training steps
        self.num_training_steps = (
            len(train_loader) * config.NUM_EPOCHS // config.GRADIENT_ACCUMULATION_STEPS
        )
        self.num_warmup_steps = int(self.num_training_steps * config.WARMUP_RATIO)
        
        # Setup learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps
        )
        
        # Setup gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Tracking variables
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_mse': [],
            'learning_rates': []
        }
        
        print(f"Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Mixed precision: {self.use_amp}")
        print(f"  Total training steps: {self.num_training_steps}")
        print(f"  Warmup steps: {self.num_warmup_steps}")
        print(f"  Gradient accumulation: {config.GRADIENT_ACCUMULATION_STEPS}")
    
    def train_epoch(self, epoch):
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{config.NUM_EPOCHS}"
        )
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    loss, predictions = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    # Scale loss for gradient accumulation
                    loss = loss / config.GRADIENT_ACCUMULATION_STEPS
            else:
                loss, predictions = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = loss / config.GRADIENT_ACCUMULATION_STEPS
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        config.MAX_GRAD_NORM
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        config.MAX_GRAD_NORM
                    )
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Track loss (unscaled)
            total_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
            num_batches += 1
            
            # Update progress bar
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Logging
            if self.global_step % config.LOGGING_STEPS == 0:
                self.training_history['train_loss'].append(avg_loss)
                self.training_history['learning_rates'].append(
                    self.scheduler.get_last_lr()[0]
                )
        
        return total_loss / num_batches
    
    def evaluate(self):
        """
        Evaluate model on validation set.
        
        Returns:
            Dict with validation metrics
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                loss, predictions = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        
        mse = np.mean((predictions - labels) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - labels))
        
        # Calculate correlation
        correlation = np.corrcoef(predictions, labels)[0, 1]
        
        metrics = {
            'loss': avg_loss,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'correlation': correlation
        }
        
        return metrics
    
    def train(self):
        """
        Full training loop.
        """
        print("\n" + "=" * 60)
        print("Starting FinBERT Training")
        print("=" * 60 + "\n")
        
        for epoch in range(1, config.NUM_EPOCHS + 1):
            # Train epoch
            train_loss = self.train_epoch(epoch)
            
            # Evaluate
            val_metrics = self.evaluate()
            
            print(f"\nEpoch {epoch} Results:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f}")
            print(f"  Val MSE:    {val_metrics['mse']:.4f}")
            print(f"  Val RMSE:   {val_metrics['rmse']:.4f}")
            print(f"  Val MAE:    {val_metrics['mae']:.4f}")
            print(f"  Val Corr:   {val_metrics['correlation']:.4f}")
            
            # Save metrics
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['val_mse'].append(val_metrics['mse'])
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_model')
                print(f"  ✓ New best model saved!")
            
            # Save checkpoint every epoch
            self.save_checkpoint(f'checkpoint_epoch_{epoch}')
            
            print()
        
        print("=" * 60)
        print("Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("=" * 60 + "\n")
        
        # Save training history
        self.save_training_history()
    
    def save_checkpoint(self, name):
        """
        Save model checkpoint.
        
        Args:
            name: Checkpoint name (without extension)
        """
        checkpoint_dir = config.FINBERT_MODEL_DIR / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        
        # Save optimizer and scheduler states
        torch.save({
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss
        }, checkpoint_dir / 'training_state.pt')
    
    def save_training_history(self):
        """
        Save training history to JSON.
        """
        history_path = config.FINBERT_MODEL_DIR / 'training_history.json'
        history_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        history = {
            key: [float(x) for x in values]
            for key, values in self.training_history.items()
        }
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Training history saved to {history_path}")


def main():
    """
    Main training function.
    """
    print("Initializing FinBERT training...")
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    # Create directories
    config.FINBERT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    train_dataset, val_dataset, tokenizer = load_finbert_data()
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset)
    
    # Create model
    print("\nCreating model...")
    model = create_finbert_model()
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = FinBERTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # Train
    trainer.train()
    
    # Save tokenizer for inference
    tokenizer_path = config.FINBERT_MODEL_DIR / 'tokenizer'
    tokenizer.save_pretrained(tokenizer_path)
    print(f"\nTokenizer saved to {tokenizer_path}")
    
    print("\n✓ Training pipeline complete!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        raise

