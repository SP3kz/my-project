from flask import Blueprint, jsonify, request
from app.services.model_service import ModelService
from app.middleware.auth_middleware import token_required

# Create blueprint
model_bp = Blueprint('model', __name__, url_prefix='/api/model')
model_service = ModelService()

@model_bp.route('/info', methods=['GET'])
@token_required
def get_model_info(current_user):
    """
    Get information about the currently loaded model
    ---
    tags:
      - Model
    security:
      - Bearer: []
    responses:
      200:
        description: Model information retrieved successfully
      500:
        description: Server error
    """
    try:
        model_info = model_service.get_model_info()
        return jsonify(model_info), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@model_bp.route('/train', methods=['POST'])
@token_required
def train_model(current_user):
    """
    Train a new model with the provided parameters
    ---
    tags:
      - Model
    security:
      - Bearer: []
    parameters:
      - in: body
        name: body
        schema:
          type: object
          required:
            - dataset_id
          properties:
            dataset_id:
              type: string
            epochs:
              type: integer
              default: 10
            batch_size:
              type: integer
              default: 32
            learning_rate:
              type: number
              format: float
              default: 0.001
            hidden_size:
              type: integer
              default: 128
    responses:
      200:
        description: Model trained successfully
      400:
        description: Invalid request parameters
      500:
        description: Server error
    """
    try:
        data = request.get_json()
        
        # Validate required parameters
        if 'dataset_id' not in data:
            return jsonify({'error': 'dataset_id is required'}), 400
        
        # Extract training parameters with defaults
        training_params = {
            'dataset_id': data['dataset_id'],
            'epochs': data.get('epochs', 10),
            'batch_size': data.get('batch_size', 32),
            'learning_rate': data.get('learning_rate', 0.001),
            'hidden_size': data.get('hidden_size', 128)
        }
        
        # Start training
        result = model_service.train_model(training_params)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@model_bp.route('/predict', methods=['POST'])
@token_required
def predict(current_user):
    """
    Make predictions using the trained model
    ---
    tags:
      - Model
    security:
      - Bearer: []
    parameters:
      - in: body
        name: body
        schema:
          type: object
          required:
            - input_data
          properties:
            input_data:
              type: array
              items:
                type: number
    responses:
      200:
        description: Prediction successful
      400:
        description: Invalid input data
      404:
        description: Model not found or not trained
      500:
        description: Server error
    """
    try:
        data = request.get_json()
        
        # Validate input data
        if 'input_data' not in data:
            return jsonify({'error': 'input_data is required'}), 400
        
        # Make prediction
        prediction = model_service.predict(data['input_data'])
        return jsonify({'prediction': prediction}), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({'error': 'Model not found. Please train a model first.'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@model_bp.route('/optimize', methods=['POST'])
@token_required
def optimize_model(current_user):
    """
    Optimize the trained model for inference
    ---
    tags:
      - Model
    security:
      - Bearer: []
    responses:
      200:
        description: Model optimized successfully
      404:
        description: Model not found or not trained
      500:
        description: Server error
    """
    try:
        result = model_service.optimize_model()
        return jsonify(result), 200
    except FileNotFoundError as e:
        return jsonify({'error': 'Model not found. Please train a model first.'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500 