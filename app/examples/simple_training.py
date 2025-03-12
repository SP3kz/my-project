import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Add the parent directory to the path so we can import from app
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.models import SimpleNN
from app.utils import normalize_data, numpy_to_tensor, tensor_to_numpy, split_data


def generate_synthetic_data(n_samples=1000, n_features=20, n_classes=2):
    """Generate synthetic data for classification."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=5,
        n_classes=n_classes,
        random_state=42
    )
    
    # Convert to one-hot encoding
    y_onehot = np.zeros((n_samples, n_classes))
    y_onehot[np.arange(n_samples), y] = 1
    
    return X, y_onehot


def train_model(model, train_data, train_labels, val_data, val_labels, 
                epochs=100, batch_size=32, lr=0.001):
    """Train the PyTorch model."""
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Convert data to tensors
    X_train = torch.FloatTensor(train_data)
    y_train = torch.FloatTensor(train_labels)
    X_val = torch.FloatTensor(val_data)
    y_val = torch.FloatTensor(val_labels)
    
    # Training history
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        
        # Mini-batch training
        permutation = torch.randperm(X_train.size()[0])
        total_loss = 0
        
        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / (X_train.size()[0] // batch_size)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            val_losses.append(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses


def evaluate_model(model, test_data, test_labels):
    """Evaluate the model on test data."""
    model.eval()
    X_test = torch.FloatTensor(test_data)
    y_test = torch.FloatTensor(test_labels)
    
    with torch.no_grad():
        outputs = model(X_test)
        
        # For classification, get the predicted class
        _, predicted = torch.max(outputs, 1)
        _, actual = torch.max(y_test, 1)
        
        # Calculate accuracy
        accuracy = (predicted == actual).sum().item() / len(actual)
        
    return accuracy, outputs


def plot_training_history(train_losses, val_losses):
    """Plot the training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()


def main():
    # Parameters
    input_size = 20
    hidden_size = 64
    output_size = 2
    epochs = 100
    batch_size = 32
    learning_rate = 0.001
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    X, y = generate_synthetic_data(n_samples=1000, n_features=input_size, n_classes=output_size)
    
    # Normalize data
    X = normalize_data(X)
    
    # Split data
    print("Splitting data into train, validation, and test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Create model
    print("Creating model...")
    model = SimpleNN(input_size, hidden_size, output_size)
    
    # Train model
    print("Training model...")
    train_losses, val_losses = train_model(
        model, X_train, y_train, X_val, y_val, 
        epochs=epochs, batch_size=batch_size, lr=learning_rate
    )
    
    # Evaluate model
    print("Evaluating model...")
    accuracy, _ = evaluate_model(model, X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Plot training history
    print("Plotting training history...")
    plot_training_history(train_losses, val_losses)
    
    # Save model
    print("Saving model...")
    model_path = f"models/model_{input_size}_{hidden_size}_{output_size}.pt"
    model.save(model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main() 