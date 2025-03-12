import requests
import numpy as np
import json
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# API endpoint
API_URL = "http://localhost:8000"

def test_health():
    """Test the health check endpoint."""
    response = requests.get(f"{API_URL}/health")
    print(f"Health check status code: {response.status_code}")
    print(f"Health check response: {response.json()}")
    return response.status_code == 200


def test_model_info():
    """Test the model info endpoint."""
    response = requests.get(f"{API_URL}/model/info")
    print(f"Model info status code: {response.status_code}")
    print(f"Model info response: {response.json()}")
    return response.status_code == 200


def test_load_model():
    """Test the load model endpoint."""
    model_config = {
        "input_size": 20,
        "hidden_size": 64,
        "output_size": 2
    }
    
    response = requests.post(f"{API_URL}/model/load", json=model_config)
    print(f"Load model status code: {response.status_code}")
    print(f"Load model response: {response.json()}")
    return response.status_code == 200


def test_prediction():
    """Test the prediction endpoint."""
    # Generate some test data
    X, _ = make_classification(
        n_samples=5,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    # Prepare request data
    request_data = {
        "features": X.tolist()
    }
    
    # Make prediction request
    response = requests.post(f"{API_URL}/predict", json=request_data)
    print(f"Prediction status code: {response.status_code}")
    
    if response.status_code == 200:
        predictions = response.json()["predictions"]
        print(f"Received {len(predictions)} predictions")
        print(f"First prediction: {predictions[0]}")
        return True
    else:
        print(f"Prediction error: {response.text}")
        return False


def test_performance(n_requests=100, batch_size=10):
    """Test the API performance."""
    # Generate test data
    X, _ = make_classification(
        n_samples=batch_size,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    # Prepare request data
    request_data = {
        "features": X.tolist()
    }
    
    # Measure response times
    response_times = []
    
    for i in range(n_requests):
        start_time = time.time()
        response = requests.post(f"{API_URL}/predict", json=request_data)
        end_time = time.time()
        
        if response.status_code == 200:
            response_time = end_time - start_time
            response_times.append(response_time)
            
            if (i + 1) % 10 == 0:
                print(f"Completed {i+1}/{n_requests} requests")
    
    # Calculate statistics
    avg_time = np.mean(response_times)
    min_time = np.min(response_times)
    max_time = np.max(response_times)
    p95_time = np.percentile(response_times, 95)
    
    print(f"\nPerformance Results:")
    print(f"Average response time: {avg_time:.4f} seconds")
    print(f"Minimum response time: {min_time:.4f} seconds")
    print(f"Maximum response time: {max_time:.4f} seconds")
    print(f"95th percentile response time: {p95_time:.4f} seconds")
    print(f"Requests per second: {1/avg_time:.2f}")
    
    # Plot response times
    plt.figure(figsize=(10, 6))
    plt.plot(response_times)
    plt.axhline(y=avg_time, color='r', linestyle='-', label=f'Average: {avg_time:.4f}s')
    plt.axhline(y=p95_time, color='g', linestyle='--', label=f'95th percentile: {p95_time:.4f}s')
    plt.xlabel('Request Number')
    plt.ylabel('Response Time (seconds)')
    plt.title('API Response Times')
    plt.legend()
    plt.savefig('api_performance.png')
    plt.close()
    
    return response_times


def main():
    """Run all tests."""
    print("Testing API...")
    
    # Test health check
    print("\n1. Testing health check endpoint...")
    if not test_health():
        print("Health check failed. Make sure the API is running.")
        return
    
    # Test model info
    print("\n2. Testing model info endpoint...")
    test_model_info()
    
    # Test load model
    print("\n3. Testing load model endpoint...")
    if not test_load_model():
        print("Failed to load model.")
        return
    
    # Test prediction
    print("\n4. Testing prediction endpoint...")
    if not test_prediction():
        print("Prediction test failed.")
        return
    
    # Test performance
    print("\n5. Testing API performance...")
    test_performance(n_requests=50, batch_size=5)
    
    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    main() 