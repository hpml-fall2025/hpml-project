"""
CNN-based volatility forecasting model using FinBERT embeddings
Following the architecture from the paper but with FinBERT instead of Word2Vec
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class FinBERTCNN(nn.Module):
    """
    FinBERT + CNN architecture for volatility forecasting
    
    Architecture:
        1. FinBERT embeddings (frozen)
        2. CNN with multiple filter sizes (unigram, bigram, trigram)
        3. Global max pooling
        4. Fully connected layers
        5. RV prediction (single value)
    """
    
    def __init__(self, config):
        super(FinBERTCNN, self).__init__()
        
        self.config = config
        
        # FinBERT for embeddings (frozen)
        self.finbert = AutoModel.from_pretrained(config.finbert_model)
        # Freeze FinBERT parameters (we only use it for embeddings)
        for param in self.finbert.parameters():
            param.requires_grad = False
        
        # CNN layers - one for each filter size
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=config.embedding_dim,  # 768 for FinBERT
                out_channels=config.num_filters,
                kernel_size=fs
            )
            for fs in config.filter_sizes  # [1, 2, 3]
        ])
        
        # Total features after concatenating all filter outputs
        total_filters = len(config.filter_sizes) * config.num_filters
        
        # Fully connected layers
        self.fc1 = nn.Linear(total_filters, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, 1)  # Output: single RV value
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Apply L2 regularization via weight_decay in optimizer
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            
        Returns:
            rv_pred: (batch_size, 1) - predicted realized volatility
        """
        # Get FinBERT embeddings
        with torch.no_grad():  # Don't compute gradients for FinBERT
            outputs = self.finbert(input_ids=input_ids, attention_mask=attention_mask)
            # Use last hidden state: (batch_size, seq_len, 768)
            embeddings = outputs.last_hidden_state
        
        # Transpose for Conv1d: (batch_size, embedding_dim, seq_len)
        # Conv1d expects (batch, channels, length)
        embeddings = embeddings.transpose(1, 2)
        
        # Apply convolutions and max pooling
        conv_outputs = []
        for conv in self.convs:
            # Apply convolution + ReLU: (batch_size, num_filters, seq_len - filter_size + 1)
            conv_out = F.relu(conv(embeddings))
            # Global max pooling: (batch_size, num_filters)
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)
        
        # Concatenate all filter outputs: (batch_size, total_filters)
        combined = torch.cat(conv_outputs, dim=1)
        
        # Apply dropout
        combined = self.dropout(combined)
        
        # Fully connected layers
        hidden = F.relu(self.fc1(combined))
        hidden = self.dropout(hidden)
        rv_pred = self.fc2(hidden)  # No activation (regression output)
        
        return rv_pred
    
    def predict(self, input_ids, attention_mask):
        """
        Prediction method (calls forward in eval mode)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(input_ids, attention_mask)


def create_model(config):
    """
    Factory function to create model instance
    
    Args:
        config: Configuration object
        
    Returns:
        model: FinBERTCNN instance
    """
    model = FinBERTCNN(config)
    model = model.to(config.device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen FinBERT parameters: {total_params - trainable_params:,}")
    
    return model
