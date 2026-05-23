import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from src.utils import logger

# Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class StockDataset(Dataset):
    """
    Sliding window dataset for sequence models.
    """
    def __init__(self, x_data, y_data, lookback, horizon):
        """
        Args:
            x_data: Numpy array of shape (N, num_features)
            y_data: Numpy array of shape (N,) containing the target (e.g., Close price)
            lookback: Number of past time steps to look back (input sequence length)
            horizon: Number of future time steps to predict (output sequence length)
        """
        self.x = torch.tensor(x_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32)
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self):
        return len(self.x) - self.lookback - self.horizon + 1

    def __getitem__(self, idx):
        # Input sequence: shape (lookback, features)
        x_seq = self.x[idx : idx + self.lookback]
        # Target sequence: shape (horizon,)
        y_seq = self.y[idx + self.lookback : idx + self.lookback + self.horizon]
        return x_seq, y_seq

# ----------------- Models -----------------

class LSTMModel(nn.Module):
    """
    Standard PyTorch LSTM model for multi-horizon forecasting.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Linear layer to map the final hidden state to the future horizon
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, lookback, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use the last hidden state of the top LSTM layer
        # h_n shape: (num_layers, batch_size, hidden_dim)
        out = self.fc(h_n[-1])
        # out shape: (batch_size, output_dim)
        return out

class GRUModel(nn.Module):
    """
    Standard PyTorch GRU model for multi-horizon forecasting.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, lookback, input_dim)
        gru_out, h_n = self.gru(x)
        # Use the last hidden state
        # h_n shape: (num_layers, batch_size, hidden_dim)
        out = self.fc(h_n[-1])
        return out

# ----------------- Deep Learning Trainer -----------------

class PyTorchTrainer:
    """
    Handles data scaling, sequence creation, fitting and evaluating LSTM and GRU models.
    """
    def __init__(self, lookback=20, horizon=5, hidden_dim=64, num_layers=2, 
                 lr=0.001, batch_size=32, epochs=50, patience=10, device=None):
        set_seed()
        self.lookback = lookback
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Trainer configured. Running on device: {self.device}")
        
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None

    def prep_dataloaders(self, train_df, val_df, feature_cols, target_col):
        """
        Extracts matrices, fits scaling transformations, and generates PyTorch Dataloaders.
        """
        # Fit scalers on training data
        train_x_raw = train_df[feature_cols].values
        train_y_raw = train_df[target_col].values.reshape(-1, 1)
        
        val_x_raw = val_df[feature_cols].values
        val_y_raw = val_df[target_col].values.reshape(-1, 1)
        
        train_x_scaled = self.scaler_x.fit_transform(train_x_raw)
        train_y_scaled = self.scaler_y.fit_transform(train_y_raw).flatten()
        
        val_x_scaled = self.scaler_x.transform(val_x_raw)
        val_y_scaled = self.scaler_y.transform(val_y_raw).flatten()
        
        # Create Datasets
        train_dataset = StockDataset(train_x_scaled, train_y_scaled, self.lookback, self.horizon)
        val_dataset = StockDataset(val_x_scaled, val_y_scaled, self.lookback, self.horizon)
        
        # Create Dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        
        logger.info(f"Data loaders created. Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        return train_loader, val_loader

    def fit(self, train_df, val_df, feature_cols, target_col, model_type='lstm', save_path=None):
        """
        Trains the selected model architecture with early stopping.
        """
        train_loader, val_loader = self.prep_dataloaders(train_df, val_df, feature_cols, target_col)
        
        input_dim = len(feature_cols)
        output_dim = self.horizon
        
        # Initialize selected architecture
        if model_type.lower() == 'lstm':
            self.model = LSTMModel(input_dim, self.hidden_dim, self.num_layers, output_dim).to(self.device)
        elif model_type.lower() == 'gru':
            self.model = GRUModel(input_dim, self.hidden_dim, self.num_layers, output_dim).to(self.device)
        else:
            raise ValueError(f"Unknown architecture: {model_type}")
            
        logger.info(f"Model Architecture:\n{self.model}")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_weights = None
        
        for epoch in range(1, self.epochs + 1):
            # Training Phase
            self.model.train()
            train_losses = []
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                
            # Validation Phase
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    preds = self.model(batch_x)
                    loss = criterion(preds, batch_y)
                    val_losses.append(loss.item())
                    
            epoch_train_loss = np.mean(train_losses)
            epoch_val_loss = np.mean(val_losses)
            
            # Print epoch logs every 5 epochs, or at start/end
            if epoch % 5 == 0 or epoch == 1 or epoch == self.epochs:
                logger.info(f"Epoch {epoch:02d}/{self.epochs} - Train MSE: {epoch_train_loss:.6f} - Val MSE: {epoch_val_loss:.6f}")
            
            # Early Stopping check
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_no_improve = 0
                best_weights = self.model.state_dict().copy()
                if save_path:
                    torch.save(best_weights, save_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.payout_check_patience():
                    logger.info(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.6f}")
                    break
                    
        # Load best weights
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
            logger.info("Best model weights loaded back to model.")
            
        return best_val_loss

    def payout_check_patience(self):
        return self.patience

    def predict(self, test_df, feature_cols):
        """
        Generates predictions on test data using rolling sequences.
        """
        if self.model is None:
            raise ValueError("Trainer model is not fitted yet.")
            
        self.model.eval()
        test_x_raw = test_df[feature_cols].values
        test_x_scaled = self.scaler_x.transform(test_x_raw)
        
        # We need lookback days prior to the prediction start to make predictions.
        # So we can feed sequences directly.
        preds_list = []
        
        # Sliding sequence prediction over test dataset
        dataset = StockDataset(
            test_x_scaled, 
            np.zeros(len(test_x_scaled)), # targets dummy placeholder
            self.lookback, 
            self.horizon
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                pred_scaled = self.model(batch_x).cpu().numpy() # shape (1, horizon)
                # Inverse scale target predictions
                pred_inverse = self.scaler_y.inverse_transform(pred_scaled).flatten()
                preds_list.append(pred_inverse)
                
        # Return predicted outputs as an array of shape (N_samples, horizon)
        return np.array(preds_list)
