import numpy as np
import torch
from typing import Tuple, List, Dict, Any, Union


def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Normalize data to have zero mean and unit variance.
    
    Args:
        data: Input data as numpy array
        
    Returns:
        Normalized data
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    # Avoid division by zero
    std = np.where(std == 0, 1, std)
    return (data - mean) / std


def numpy_to_tensor(data: np.ndarray, device: str = 'cpu') -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.
    
    Args:
        data: Input numpy array
        device: Target device ('cpu' or 'cuda')
        
    Returns:
        PyTorch tensor
    """
    return torch.tensor(data, dtype=torch.float32).to(device)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy array.
    
    Args:
        tensor: Input PyTorch tensor
        
    Returns:
        Numpy array
    """
    return tensor.detach().cpu().numpy()


def split_data(data: np.ndarray, labels: np.ndarray, 
               test_size: float = 0.2, 
               val_size: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training, validation and test sets.
    
    Args:
        data: Input data
        labels: Input labels
        test_size: Proportion of data to use for testing
        val_size: Proportion of data to use for validation
        
    Returns:
        Tuple of (train_data, val_data, test_data, train_labels, val_labels, test_labels)
    """
    n_samples = len(data)
    indices = np.random.permutation(n_samples)
    
    test_count = int(test_size * n_samples)
    val_count = int(val_size * n_samples)
    
    test_indices = indices[:test_count]
    val_indices = indices[test_count:test_count + val_count]
    train_indices = indices[test_count + val_count:]
    
    train_data = data[train_indices]
    train_labels = labels[train_indices]
    
    val_data = data[val_indices]
    val_labels = labels[val_indices]
    
    test_data = data[test_indices]
    test_labels = labels[test_indices]
    
    return train_data, val_data, test_data, train_labels, val_labels, test_labels 