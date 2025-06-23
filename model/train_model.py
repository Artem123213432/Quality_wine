import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
from utils import (
    load_and_preprocess_data, 
    create_data_loaders, 
    evaluate_model, 
    plot_predictions,
    plot_error_histogram,
    plot_residuals
)


class WineQualityNN(nn.Module):
    """Neural network for wine quality prediction"""
    
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32):
        super(WineQualityNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1)
        )
        
    def forward(self, x):
        return self.layers(x)


def train_model(model, train_loader, test_loader, criterion, optimizer, 
                epochs=300, device='cpu', print_every=50):
    """
    Train neural network model
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        test_loader: Test data loader
        criterion: Loss function
        optimizer: Optimization algorithm
        epochs: Number of training epochs
        device: Computation device (CPU/GPU)
        print_every: Progress printing interval
        
    Returns:
        Dict with training history and evaluation metrics
    """
    model.to(device)
    start_time = time.time()
    
    history = {
        'train_loss': [],
        'test_loss': []
    }
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for batch in train_loader:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        model.eval()
        test_losses = []
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(device)
                targets = batch['targets'].to(device)
                
                outputs = model(features)
                loss = criterion(outputs, targets)
                test_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        avg_test_loss = np.mean(test_losses)
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)
        
        if (epoch + 1) % print_every == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Test Loss: {avg_test_loss:.4f}')
    
    training_time = time.time() - start_time
    print(f'\nTraining completed in {training_time:.2f} seconds')
    
    results = evaluate_model(model, test_loader, device)
    print(f"\nTest metrics:")
    print(f"R² Score: {results['r2_score']:.3f}")
    print(f"MAE: {results['mae']:.3f}")
    print(f"MSE: {results['mse']:.3f}")
    
    return {
        'history': history,
        'metrics': results,
        'model': model,
        'training_time': training_time
    }


def plot_learning_curves(history, save_path=None):
    """
    Plot learning curves
    
    Args:
        history: Model training history
        save_path: Path to save the plot image
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['test_loss'], label='Validation Loss')
    plt.title('Loss Dynamics During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def save_metrics(metrics, save_path):
    """Save model metrics to a text file"""
    with open(save_path, 'w') as f:
        f.write(f"MSE: {metrics['mse']:.4f}\n")
        f.write(f"MAE: {metrics['mae']:.4f}\n")
        f.write(f"R²: {metrics['r2_score']:.4f}\n")


def main():
    # Data path
    data_path = '../data/winequality-red.csv'
    # Get absolute path to data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(current_dir), 'data', 'winequality-red.csv')
    
    # Create results directory
    results_dir = '../results'
    results_dir = os.path.join(os.path.dirname(current_dir), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(data_path)
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(X_train, X_test, y_train, y_test, batch_size=64)
    
    # Set computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model, loss function and optimizer
    input_dim = X_train.shape[1]
    model = WineQualityNN(input_dim=input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    results = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=300,
        device=device,
        print_every=50
    )
    
    # Save metrics
    save_metrics(
        results['metrics'],
        os.path.join(results_dir, 'metrics.txt')
    )
    
    # Plot and save visualizations
    plot_learning_curves(
        results['history'],
        save_path=os.path.join(results_dir, 'learning_curves.png')
    )
    
    y_true = results['metrics']['y_true']
    y_pred = results['metrics']['y_pred']
    
    plot_predictions(
        y_true=y_true,
        y_pred=y_pred,
        save_path=os.path.join(results_dir, 'predictions.png')
    )
    
    plot_error_histogram(
        y_true=y_true,
        y_pred=y_pred,
        save_path=os.path.join(results_dir, 'error_histogram.png')
    )
    
    plot_residuals(
        y_true=y_true,
        y_pred=y_pred,
        save_path=os.path.join(results_dir, 'residuals.png')
    )
    
    # Save model
    model_path = os.path.join(results_dir, 'wine_quality_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(results_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")


if __name__ == "__main__":
    main() 