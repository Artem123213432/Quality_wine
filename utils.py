import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from torch.utils.data import Dataset, DataLoader


class WineDataset(Dataset):
    """Dataset for wine quality data"""
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'targets': self.targets[idx]
        }


def load_and_preprocess_data(data_path, test_size=0.2, random_state=42):
    """
    Load and preprocess wine quality data
    
    Args:
        data_path: Path to the data file
        test_size: Proportion of test set
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # Load data
    data = pd.read_csv(data_path, sep=";")
    
    # Split features and target
    X = data.drop('quality', axis=1).values
    y = data['quality'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler


def create_data_loaders(X_train, X_test, y_train, y_test, batch_size=32):
    """
    Create PyTorch data loaders
    
    Args:
        X_train, X_test, y_train, y_test: Training and test data
        batch_size: Batch size for training
        
    Returns:
        train_loader, test_loader
    """
    # Create datasets
    train_dataset = WineDataset(X_train, y_train)
    test_dataset = WineDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate the model on test data
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Computation device
        
    Returns:
        Dict with evaluation metrics
    """
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            
            outputs = model(features)
            
            y_true.extend(targets.cpu().numpy().flatten())
            y_pred.extend(outputs.cpu().numpy().flatten())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'mse': mse,
        'mae': mae,
        'r2_score': r2
    }


def plot_predictions(y_true, y_pred, save_path=None):
    """
    Plot predicted vs actual values
    
    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Path to save the plot image
    """
    plt.figure(figsize=(10, 8))
    
    # Perfect predictions line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    
    # Actual predictions scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6, s=60, edgecolors='k')
    
    plt.xlabel('Actual Wine Quality')
    plt.ylabel('Predicted Wine Quality')
    plt.title('Predicted vs Actual Wine Quality')
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.close()


def plot_error_histogram(y_true, y_pred, save_path=None, bins=20):
    """
    Plot histogram of prediction errors
    
    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Path to save the plot image
        bins: Number of histogram bins
    """
    errors = y_pred - y_true
    
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
    plt.axvline(x=np.mean(errors), color='green', linestyle='-', linewidth=2,
               label=f'Mean Error: {np.mean(errors):.3f}')
    
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Histogram of Prediction Errors')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_residuals(y_true, y_pred, save_path=None):
    """
    Plot residuals against predicted values
    
    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Path to save the plot image
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='k')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close() 