import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from src.models.lstm_gru import StockDataset, set_seed
from src.utils import logger

class TemporalAttention(nn.Module):
    """
    Computes Bahdanau (additive) attention over encoder hidden states.
    """
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.W_query = nn.Linear(hidden_dim, hidden_dim)
        self.W_keys = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, query, keys):
        """
        Args:
            query: Final encoder hidden state. Shape: (batch_size, hidden_dim)
            keys: Encoder hidden states for all steps. Shape: (batch_size, lookback_len, hidden_dim)
            
        Returns:
            context_vector: Weighted sum of keys. Shape: (batch_size, hidden_dim)
            attention_weights: Weights for each step. Shape: (batch_size, lookback_len)
        """
        # Expand query dimensions: (batch_size, 1, hidden_dim)
        query_expanded = query.unsqueeze(1)
        
        # Compute alignment scores
        # scores shape: (batch_size, lookback_len, 1)
        scores = self.v(torch.tanh(self.W_query(query_expanded) + self.W_keys(keys)))
        
        # Remove trailing dimension and compute weights
        # weights shape: (batch_size, lookback_len)
        attention_weights = F.softmax(scores.squeeze(-1), dim=1)
        
        # Apply weights to keys
        # keys shape: (batch_size, lookback_len, hidden_dim)
        # weights unsqueezed: (batch_size, lookback_len, 1)
        # context shape: (batch_size, hidden_dim)
        context_vector = torch.sum(keys * attention_weights.unsqueeze(-1), dim=1)
        
        return context_vector, attention_weights

class AttentionLSTMModel(nn.Module):
    """
    Sequence-to-Sequence Attention-augmented LSTM for stock forecasting.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(AttentionLSTMModel, self).__init__()
        set_seed()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.attention = TemporalAttention(hidden_dim)
        
        # Fully connected layer combining context vector and final hidden state
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        # Placeholders to save weights during inference
        self.latest_attention_weights = None

    def forward(self, x):
        # x shape: (batch_size, lookback, input_dim)
        
        # lstm_out shape (keys): (batch_size, lookback, hidden_dim)
        # h_n shape: (num_layers, batch_size, hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take final hidden state of encoder (query)
        query = h_n[-1] # shape: (batch_size, hidden_dim)
        
        # Compute context vector and attention weights
        context, attn_weights = self.attention(query, lstm_out)
        
        # Store weights for plotting during visualization steps
        self.latest_attention_weights = attn_weights
        
        # Concatenate context vector and query
        # combined shape: (batch_size, hidden_dim * 2)
        combined = torch.cat((query, context), dim=1)
        
        # Project to target prediction horizon (output_dim)
        out = self.fc(combined)
        return out

class AttentionLSTMHelper:
    """
    Wrapper helper to handle training and attention analysis of the Attention-LSTM model.
    """
    def __init__(self, trainer):
        """
        Args:
            trainer: Instance of PyTorchTrainer (to reuse scalers and parameters)
        """
        self.trainer = trainer
        self.model = None

    def fit(self, train_df, val_df, feature_cols, target_col, save_path=None):
        """
        Trains the Attention-LSTM model.
        """
        train_loader, val_loader = self.trainer.prep_dataloaders(train_df, val_df, feature_cols, target_col)
        
        input_dim = len(feature_cols)
        output_dim = self.trainer.horizon
        
        self.model = AttentionLSTMModel(
            input_dim=input_dim,
            hidden_dim=self.trainer.hidden_dim,
            num_layers=self.trainer.num_layers,
            output_dim=output_dim
        ).to(self.trainer.device)
        
        logger.info(f"Attention-LSTM Model Architecture:\n{self.model}")
        
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
                logger.info(f"Epoch {epoch:02d}/{self.trainer.epochs} - Attention-LSTM Train MSE: {epoch_train_loss:.6f} - Val MSE: {epoch_val_loss:.6f}")
                
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_no_improve = 0
                best_weights = self.model.state_dict().copy()
                if save_path:
                    torch.save(best_weights, save_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.trainer.patience:
                    logger.info(f"Attention-LSTM Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.6f}")
                    break
                    
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
            logger.info("Attention-LSTM best weights loaded back.")
            
        # Give trainer access to model for inference standard routines
        self.trainer.model = self.model
        return best_val_loss

    def extract_attention_weights(self, test_df, feature_cols):
        """
        Runs inference and extracts historical attention weight map.
        
        Returns:
            mean_weights: Average attention distribution over all test sequences (shape: lookback)
            full_weights: List of raw weights for each sequence
        """
        if self.model is None:
            raise ValueError("Model has not been trained.")
            
        self.model.eval()
        test_x_raw = test_df[feature_cols].values
        test_x_scaled = self.trainer.scaler_x.transform(test_x_raw)
        
        dataset = StockDataset(test_x_scaled, np.zeros(len(test_x_scaled)), self.trainer.lookback, self.trainer.horizon)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        all_weights = []
        
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.trainer.device)
                _ = self.model(batch_x)
                # Fetch saved weights
                weights = self.model.latest_attention_weights.cpu().numpy().flatten()
                all_weights.append(weights)
                
        all_weights = np.array(all_weights)
        mean_weights = np.mean(all_weights, axis=0)
        
        logger.info(f"Attention weight extraction completed. Lookback Attention shape: {mean_weights.shape}")
        return mean_weights, all_weights
