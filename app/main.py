import os
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import logging
import time

from app.models import SimpleNN
from app.utils import normalize_data, numpy_to_tensor, tensor_to_numpy
from app.utils import optimize_model_for_inference, batch_predictions, cache_model_weights, load_cached_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create the FastAPI app
app = FastAPI(
    title="ML Project API",
    description="API for machine learning predictions using PyTorch",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create directories if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("models/cache", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Global variable to store the model
model = None
model_info = {
    "loaded": False,
    "input_size": 0,
    "hidden_size": 0,
    "output_size": 0,
    "optimized": False,
    "cached": False
}

# Pydantic models for request/response validation
class PredictionInput(BaseModel):
    features: List[List[float]]

class PredictionOutput(BaseModel):
    predictions: List[List[float]]
    processing_time: float

class ModelInfo(BaseModel):
    loaded: bool
    input_size: int
    hidden_size: int
    output_size: int
    optimized: bool
    cached: bool

class TrainingConfig(BaseModel):
    input_size: int
    hidden_size: int
    output_size: int
    optimize: bool = True
    cache: bool = True


@app.get("/")
async def root():
    """Root endpoint that returns a welcome message."""
    return {"message": "Welcome to the ML Project API"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the currently loaded model."""
    return model_info


@app.post("/model/load")
async def load_model(config: TrainingConfig, background_tasks: BackgroundTasks):
    """
    Load a pre-trained model.
    
    If the model file doesn't exist, a new model will be initialized.
    """
    global model, model_info
    
    model_path = f"models/model_{config.input_size}_{config.hidden_size}_{config.output_size}.pt"
    cache_path = f"models/cache/model_{config.input_size}_{config.hidden_size}_{config.output_size}.pt"
    
    try:
        start_time = time.time()
        
        if os.path.exists(cache_path) and config.cache:
            # Load from cache
            logger.info(f"Loading model from cache: {cache_path}")
            model_args = {
                "input_size": config.input_size,
                "hidden_size": config.hidden_size,
                "output_size": config.output_size
            }
            model = load_cached_model(SimpleNN, cache_path, model_args)
            logger.info(f"Model loaded from cache in {time.time() - start_time:.2f} seconds")
            cached = True
        elif os.path.exists(model_path):
            # Load from model file
            logger.info(f"Loading model from {model_path}")
            model = SimpleNN.load(
                model_path, 
                config.input_size, 
                config.hidden_size, 
                config.output_size
            )
            logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
            cached = False
        else:
            # Initialize new model
            logger.info(f"Model file not found. Initializing new model.")
            model = SimpleNN(
                config.input_size, 
                config.hidden_size, 
                config.output_size
            )
            logger.info("New model initialized")
            cached = False
        
        # Optimize model if requested
        optimized = False
        if config.optimize:
            logger.info("Optimizing model for inference")
            model = optimize_model_for_inference(model)
            optimized = True
            logger.info("Model optimized")
        
        # Cache model if requested
        if config.cache and not cached:
            logger.info("Caching model weights")
            background_tasks.add_task(cache_model_weights, model, "models/cache")
            cached = True
            
        model_info = {
            "loaded": True,
            "input_size": config.input_size,
            "hidden_size": config.hidden_size,
            "output_size": config.output_size,
            "optimized": optimized,
            "cached": cached
        }
        
        load_time = time.time() - start_time
        logger.info(f"Model loading completed in {load_time:.2f} seconds")
        
        return {
            "message": "Model loaded successfully", 
            "initialized_new": not os.path.exists(model_path),
            "optimized": optimized,
            "cached": cached,
            "load_time": load_time
        }
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Make predictions using the loaded model.
    """
    global model, model_info
    
    if not model_info["loaded"] or model is None:
        raise HTTPException(status_code=400, detail="No model loaded. Please load a model first.")
    
    try:
        start_time = time.time()
        
        # Convert input data to numpy array
        features = np.array(input_data.features, dtype=np.float32)
        
        # Normalize data
        features = normalize_data(features)
        
        # Make predictions using batching for efficiency
        batch_size = min(32, len(features))  # Use smaller batch size for small inputs
        predictions = batch_predictions(model, features, batch_size=batch_size)
        
        # Convert predictions to list
        predictions_list = predictions.tolist()
        
        processing_time = time.time() - start_time
        logger.info(f"Prediction completed in {processing_time:.4f} seconds for {len(features)} samples")
        
        return {"predictions": predictions_list, "processing_time": processing_time}
    
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error making predictions: {str(e)}")


@app.post("/upload/data")
async def upload_data(file: UploadFile = File(...)):
    """
    Upload a CSV file with data for processing.
    """
    try:
        start_time = time.time()
        
        contents = await file.read()
        file_path = f"data/{file.filename}"
        
        with open(file_path, "wb") as f:
            f.write(contents)
            
        processing_time = time.time() - start_time
        logger.info(f"File {file.filename} uploaded in {processing_time:.4f} seconds")
        
        return {
            "filename": file.filename, 
            "message": "File uploaded successfully",
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/cache", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Run the FastAPI app with uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 