import torch
import torch.nn as nn
import numpy as np
import math
from src.models.lstm_gru import set_seed
from src.utils import logger

class PositionalEncoding(nn.Module):
    """
    Standard sinusoids positional encoding for sequence modeling.
    """
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]

class TimeSeriesTransformer(nn.Module):
    """
    Custom PyTorch Transformer designed specifically for numeric time-series multi-horizon forecasting.
    """
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, 
                 output_dim=5, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        set_seed()
        
        # Project raw input features to embedding dimension d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Custom Transformer Encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output decoder projections (aggregates features over lookback length)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
        # Projects outputs to the multi-horizon output_dim
        self.output_projection = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, lookback_len, input_dim)
        
        # Project to d_model: shape (batch_size, lookback_len, d_model)
        x_proj = self.input_projection(x)
        
        # Add positional encoding
        x_pe = self.pos_encoder(x_proj)
        
        # Pass through Transformer encoder: shape (batch_size, lookback_len, d_model)
        encoded = self.transformer_encoder(x_pe)
        
        # Take the mean over the lookback sequence length to summarize temporal context
        # (Alternatively, we can use the last timestep or a pooling mechanism)
        pooled = torch.mean(encoded, dim=1) # shape: (batch_size, d_model)
        
        # Project to output forecast horizon
        out = self.output_projection(pooled) # shape: (batch_size, output_dim)
        return out

class TransformerHelper:
    """
    Wrapper helper to handle training and evaluations of the Time-Series Transformer model.
    """
    def __init__(self, trainer, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        self.trainer = trainer
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.model = None

    def fit(self, train_df, val_df, feature_cols, target_col, save_path=None):
        """
        Trains the Time Series Transformer model.
        """
        train_loader, val_loader = self.trainer.prep_dataloaders(train_df, val_df, feature_cols, target_col)
        
        input_dim = len(feature_cols)
        output_dim = self.trainer.horizon
        
        self.model = TimeSeriesTransformer(
            input_dim=input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            output_dim=output_dim
        ).to(self.trainer.device)
        
        logger.info(f"Transformer Model Architecture:\n{self.model}")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.trainer.lr)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_weights = None
        
        for epoch in range(1, self.trainer.epochs + 1):
            self.model.train()
            train_losses = []
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.trainer.device), batch_y.to(self.trainer.device)
                
                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.trainer.device), batch_y.to(self.trainer.device)
                    preds = self.model(batch_x)
                    loss = criterion(preds, batch_y)
                    val_losses.append(loss.item())
                    
            epoch_train_loss = np.mean(train_losses)
            epoch_val_loss = np.mean(val_losses)
            
            if epoch % 5 == 0 or epoch == 1 or epoch == self.trainer.epochs:
                logger.info(f"Epoch {epoch:02d}/{self.trainer.epochs} - Transformer Train MSE: {epoch_train_loss:.6f} - Val MSE: {epoch_val_loss:.6f}")
                
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_no_improve = 0
                best_weights = self.model.state_dict().copy()
                if save_path:
                    torch.save(best_weights, save_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.trainer.patience:
                    logger.info(f"Transformer Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.6f}")
                    break
                    
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
            logger.info("Transformer best weights loaded back.")
            
        # Bind back to the main trainer
        self.trainer.model = self.model
        return best_val_loss
