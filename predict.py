import argparse
import numpy as np
import pandas as pd
import torch
import pickle
import os
import sys
from model.train_model import WineQualityNN

# Добавляем текущую директорию в путь для поиска модулей
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def load_model_and_scaler(model_path, scaler_path=None):
    """
    Load the trained model and scaler
    
    Args:
        model_path: Path to the saved model
        scaler_path: Path to the saved scaler
        
    Returns:
        model, scaler
    """
    # Load model
    input_dim = 11  # Number of features in wine quality dataset
    model = WineQualityNN(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # Load scaler if provided
    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    
    return model, scaler

def predict_single_sample(model, features, scaler=None):
    """
    Make prediction for a single sample
    
    Args:
        model: Trained model
        features: Input features (array-like)
        scaler: Scaler for normalization
        
    Returns:
        Predicted wine quality
    """
    # Convert to numpy array if not already
    features_np = np.array(features).reshape(1, -1)
    
    # Apply scaling if scaler is provided
    if scaler:
        features_np = scaler.transform(features_np)
    
    # Convert to torch tensor
    features_tensor = torch.tensor(features_np, dtype=torch.float32)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(features_tensor)
    
    return prediction.item()

def predict_from_csv(model, csv_path, scaler=None, delimiter=';', feature_cols=None):
    """
    Make predictions for wine samples in a CSV file
    
    Args:
        model: Trained model
        csv_path: Path to the CSV file with wine samples
        scaler: Scaler for normalization
        delimiter: CSV delimiter character
        feature_cols: List of feature column names
        
    Returns:
        DataFrame with predictions
    """
    # Read data
    data = pd.read_csv(csv_path, delimiter=delimiter)
    
    # Use default feature columns if not provided
    if feature_cols is None:
        # Default wine quality dataset features
        feature_cols = [
            'fixed acidity', 'volatile acidity', 'citric acid', 
            'residual sugar', 'chlorides', 'free sulfur dioxide',
            'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
        ]
    
    # Extract features
    features = data[feature_cols].values
    
    # Apply scaling if scaler is provided
    if scaler:
        features = scaler.transform(features)
    
    # Convert to torch tensor and make predictions
    features_tensor = torch.tensor(features, dtype=torch.float32)
    
    with torch.no_grad():
        predictions = model(features_tensor)
    
    # Add predictions to the dataframe
    data['predicted_quality'] = predictions.numpy().flatten()
    
    return data

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Predict wine quality')
    parser.add_argument('--model', type=str, default='results/wine_quality_model.pth',
                        help='Path to the trained model')
    parser.add_argument('--scaler', type=str, default='results/scaler.pkl',
                        help='Path to the saved scaler')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input CSV file or comma-separated feature values')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save predictions (for CSV input)')
    
    args = parser.parse_args()
    
    # Load model and scaler
    model, scaler = load_model_and_scaler(args.model, args.scaler)
    
    # Check if input is a CSV file or a single sample
    if args.input.endswith('.csv'):
        # Make predictions from CSV
        results = predict_from_csv(model, args.input, scaler)
        
        # Print summary
        print(f"\nPredictions for {len(results)} samples:")
        print(f"Average predicted quality: {results['predicted_quality'].mean():.2f}")
        print(f"Min predicted quality: {results['predicted_quality'].min():.2f}")
        print(f"Max predicted quality: {results['predicted_quality'].max():.2f}")
        
        # Save results if output path is provided
        if args.output:
            results.to_csv(args.output, index=False)
            print(f"\nPredictions saved to {args.output}")
    else:
        # Parse single sample features
        try:
            features = [float(x.strip()) for x in args.input.split(',')]
            if len(features) != 11:
                print("Error: Input must have 11 features for wine quality prediction.")
                print("The features should be in this order:")
                print("fixed acidity, volatile acidity, citric acid, residual sugar, chlorides,")
                print("free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol")
                sys.exit(1)
                
            # Make prediction
            prediction = predict_single_sample(model, features, scaler)
            
            # Print result
            print(f"\nPredicted wine quality: {prediction:.2f}")
            if prediction < 3:
                print("Quality interpretation: Poor")
            elif prediction < 5:
                print("Quality interpretation: Below Average")
            elif prediction < 7:
                print("Quality interpretation: Average to Good")
            else:
                print("Quality interpretation: Excellent")
                
        except ValueError:
            print("Error: Invalid input format. Use comma-separated values or a CSV file.")
            sys.exit(1)

if __name__ == "__main__":
    main() 