import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class WineDataset(Dataset):
    """Wine quality dataset for PyTorch DataLoader"""
    
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float32),
            'targets': torch.tensor(self.targets[idx], dtype=torch.float32)
        }


def load_and_preprocess_data(data_path, test_size=0.2, random_state=42):
    """
    Load and preprocess wine quality data
    
    Args:
        data_path: Path to CSV data file
        test_size: Proportion of test set
        random_state: Random seed for reproducibility
        
    Returns:
        Processed train/test data and scaler
    """
    wine_data = pd.read_csv(data_path, sep=';')
    
    X = wine_data.drop('quality', axis=1).values
    y = wine_data['quality'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler


def create_data_loaders(X_train, X_test, y_train, y_test, batch_size=64):
    """
    Create PyTorch data loaders
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test target values
        batch_size: Mini-batch size
        
    Returns:
        PyTorch DataLoader objects
    """
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    train_dataset = WineDataset(X_train, y_train)
    test_dataset = WineDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def evaluate_model(model, test_loader, device):
    """
    Evaluate model performance on test data
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Computation device (CPU/GPU)
        
    Returns:
        Dict with evaluation metrics and predictions
    """
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            
            outputs = model(features)
            
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())
    
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    
    return {
        'r2_score': r2,
        'mae': mae,
        'mse': mse,
        'y_true': y_true,
        'y_pred': y_pred
    }


def plot_predictions(y_true, y_pred, save_path=None):
    """
    Plot actual vs predicted values
    
    Args:
        y_true: True target values
        y_pred: Model predictions
        save_path: Path to save the plot image
    """
    plt.figure(figsize=(10, 8))
    
    plt.scatter(y_true, y_pred, alpha=0.6, color='darkblue')
    
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.title('True vs Predicted Wine Quality', fontsize=16)
    plt.xlabel('True Quality', fontsize=14)
    plt.ylabel('Predicted Quality', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    plt.annotate(f'R² = {r2:.3f}\nMAE = {mae:.3f}', 
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                 fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.close()


def plot_error_histogram(y_true, y_pred, save_path=None):
    """
    Plot histogram of prediction errors
    
    Args:
        y_true: True target values
        y_pred: Model predictions
        save_path: Path to save the plot image
    """
    errors = y_pred - y_true
    
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
    
    plt.title('Distribution of Prediction Errors', fontsize=16)
    plt.xlabel('Prediction Error (Predicted - True)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.close()


def plot_residuals(y_true, y_pred, save_path=None):
    """
    Plot residuals against predicted values
    
    Args:
        y_true: True target values
        y_pred: Model predictions
        save_path: Path to save the plot image
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, color='darkblue')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
    
    plt.title('Residuals vs Predicted Values', fontsize=16)
    plt.xlabel('Predicted Quality', fontsize=14)
    plt.ylabel('Residuals (True - Predicted)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.close() 