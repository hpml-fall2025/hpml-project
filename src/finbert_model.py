"""
FinBERT model architecture for continuous sentiment scoring.
Based on BERT with a regression head for outputting continuous sentiment values.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
import config


class FinBERT(nn.Module):
    """
    FinBERT model for continuous sentiment scoring.
    
    Architecture:
        - BERT base model (bert-base-uncased)
        - Dropout layer
        - Linear regression head (output: single continuous value)
    """
    
    def __init__(
        self,
        model_name=None,
        hidden_dropout_prob=None,
        attention_probs_dropout_prob=None,
        freeze_embeddings=False,
        freeze_encoder_layers=0
    ):
        """
        Args:
            model_name: HuggingFace model name (default: config.MODEL_NAME)
            hidden_dropout_prob: Dropout probability for hidden layers
            attention_probs_dropout_prob: Dropout probability for attention
            freeze_embeddings: Whether to freeze embedding layers
            freeze_encoder_layers: Number of encoder layers to freeze (0-12)
        """
        super(FinBERT, self).__init__()
        
        model_name = model_name or config.MODEL_NAME
        hidden_dropout_prob = hidden_dropout_prob or config.HIDDEN_DROPOUT_PROB
        attention_probs_dropout_prob = attention_probs_dropout_prob or config.ATTENTION_PROBS_DROPOUT_PROB
        
        # Load BERT configuration
        bert_config = BertConfig.from_pretrained(model_name)
        bert_config.hidden_dropout_prob = hidden_dropout_prob
        bert_config.attention_probs_dropout_prob = attention_probs_dropout_prob
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(model_name, config=bert_config)
        
        # Freeze layers if specified
        if freeze_embeddings:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
        
        if freeze_encoder_layers > 0:
            for layer in self.bert.encoder.layer[:freeze_encoder_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # Regression head for continuous sentiment output
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.regressor = nn.Linear(bert_config.hidden_size, 1)
        
        # Initialize regression head weights
        nn.init.normal_(self.regressor.weight, std=0.02)
        nn.init.zeros_(self.regressor.bias)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (batch_size, seq_length)
            attention_mask: Attention mask (batch_size, seq_length)
            labels: Ground truth sentiment scores (batch_size,) - optional
        
        Returns:
            If labels provided:
                loss, predictions
            Else:
                predictions
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output  # (batch_size, hidden_size)
        
        # Apply dropout and regression head
        pooled_output = self.dropout(pooled_output)
        predictions = self.regressor(pooled_output).squeeze(-1)  # (batch_size,)
        
        # Calculate loss if labels provided
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(predictions, labels)
            return loss, predictions
        
        return predictions
    
    def save_pretrained(self, save_directory):
        """
        Save model to directory.
        
        Args:
            save_directory: Path to save directory
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        torch.save(self.state_dict(), save_directory / "pytorch_model.bin")
        
        # Save BERT config
        self.bert.config.to_json_file(save_directory / "config.json")
        
        print(f"Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, model_directory):
        """
        Load model from directory.
        
        Args:
            model_directory: Path to saved model directory
        
        Returns:
            FinBERT model
        """
        model_directory = Path(model_directory)
        
        # Load config
        bert_config = BertConfig.from_json_file(model_directory / "config.json")
        
        # Create model instance
        model = cls(
            model_name=None,  # Will use config
            hidden_dropout_prob=bert_config.hidden_dropout_prob,
            attention_probs_dropout_prob=bert_config.attention_probs_dropout_prob
        )
        
        # Load weights
        state_dict = torch.load(
            model_directory / "pytorch_model.bin",
            map_location=torch.device('cpu')
        )
        model.load_state_dict(state_dict)
        
        print(f"Model loaded from {model_directory}")
        
        return model
    
    def get_num_parameters(self):
        """
        Get number of trainable and total parameters.
        
        Returns:
            Dict with 'trainable' and 'total' counts
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        
        return {
            'trainable': trainable,
            'total': total,
            'frozen': total - trainable
        }


def create_finbert_model(
    freeze_embeddings=None,
    freeze_encoder_layers=None
):
    """
    Create FinBERT model with configuration from config.py.
    
    Args:
        freeze_embeddings: Override config.FREEZE_EMBEDDINGS
        freeze_encoder_layers: Override config.FREEZE_ENCODER_LAYERS
    
    Returns:
        FinBERT model
    """
    freeze_embeddings = freeze_embeddings if freeze_embeddings is not None else config.FREEZE_EMBEDDINGS
    freeze_encoder_layers = freeze_encoder_layers if freeze_encoder_layers is not None else config.FREEZE_ENCODER_LAYERS
    
    model = FinBERT(
        model_name=config.MODEL_NAME,
        hidden_dropout_prob=config.HIDDEN_DROPOUT_PROB,
        attention_probs_dropout_prob=config.ATTENTION_PROBS_DROPOUT_PROB,
        freeze_embeddings=freeze_embeddings,
        freeze_encoder_layers=freeze_encoder_layers
    )
    
    # Print parameter counts
    param_counts = model.get_num_parameters()
    print(f"Model parameters:")
    print(f"  Total:     {param_counts['total']:,}")
    print(f"  Trainable: {param_counts['trainable']:,}")
    print(f"  Frozen:    {param_counts['frozen']:,}")
    
    return model


if __name__ == "__main__":
    """Test the model"""
    print("Testing FinBERT model...")
    
    # Create model
    model = create_finbert_model()
    
    # Test forward pass with dummy data
    batch_size = 4
    seq_length = 128
    
    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    dummy_attention_mask = torch.ones((batch_size, seq_length))
    dummy_labels = torch.randn(batch_size)
    
    # Forward pass with labels
    loss, predictions = model(
        input_ids=dummy_input_ids,
        attention_mask=dummy_attention_mask,
        labels=dummy_labels
    )
    
    print(f"\n✓ Model test passed!")
    print(f"  Input shape: {dummy_input_ids.shape}")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Sample predictions: {predictions[:3].detach().numpy()}")

