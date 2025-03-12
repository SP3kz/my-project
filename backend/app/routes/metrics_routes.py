from flask import Blueprint, jsonify, request
from app.services.metrics_service import MetricsService
from app.middleware.auth_middleware import token_required

# Create blueprint
metrics_bp = Blueprint('metrics', __name__, url_prefix='/api/metrics')
metrics_service = MetricsService()

@metrics_bp.route('/performance', methods=['GET'])
@token_required
def get_performance_metrics(current_user):
    """
    Get performance metrics for the specified time range
    ---
    tags:
      - Metrics
    security:
      - Bearer: []
    parameters:
      - name: start_date
        in: query
        type: string
        format: date
        required: false
        description: Start date for metrics (YYYY-MM-DD)
      - name: end_date
        in: query
        type: string
        format: date
        required: false
        description: End date for metrics (YYYY-MM-DD)
      - name: interval
        in: query
        type: string
        enum: [hourly, daily, weekly, monthly]
        required: false
        description: Interval for aggregating metrics
    responses:
      200:
        description: Performance metrics retrieved successfully
      500:
        description: Server error
    """
    try:
        # Get query parameters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        interval = request.args.get('interval', 'daily')
        
        # Get metrics
        metrics = metrics_service.get_performance_metrics(start_date, end_date, interval)
        return jsonify(metrics), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@metrics_bp.route('/errors', methods=['GET'])
@token_required
def get_error_logs(current_user):
    """
    Get error logs for the specified filters
    ---
    tags:
      - Metrics
    security:
      - Bearer: []
    parameters:
      - name: start_date
        in: query
        type: string
        format: date
        required: false
        description: Start date for logs (YYYY-MM-DD)
      - name: end_date
        in: query
        type: string
        format: date
        required: false
        description: End date for logs (YYYY-MM-DD)
      - name: error_type
        in: query
        type: string
        required: false
        description: Filter by error type
      - name: limit
        in: query
        type: integer
        required: false
        description: Maximum number of logs to return
    responses:
      200:
        description: Error logs retrieved successfully
      500:
        description: Server error
    """
    try:
        # Get query parameters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        error_type = request.args.get('error_type')
        limit = request.args.get('limit', 100, type=int)
        
        # Get error logs
        logs = metrics_service.get_error_logs(start_date, end_date, error_type, limit)
        return jsonify(logs), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@metrics_bp.route('/summary', methods=['GET'])
@token_required
def get_metrics_summary(current_user):
    """
    Get a summary of key metrics
    ---
    tags:
      - Metrics
    security:
      - Bearer: []
    responses:
      200:
        description: Metrics summary retrieved successfully
      500:
        description: Server error
    """
    try:
        # Get summary
        summary = metrics_service.get_metrics_summary()
        return jsonify(summary), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@metrics_bp.route('/log', methods=['POST'])
def log_request():
    """
    Log a request for metrics tracking (internal use)
    ---
    tags:
      - Metrics
    parameters:
      - in: body
        name: body
        schema:
          type: object
          required:
            - endpoint
            - response_time
          properties:
            endpoint:
              type: string
            response_time:
              type: number
              format: float
            status_code:
              type: integer
            error:
              type: string
    responses:
      200:
        description: Request logged successfully
      400:
        description: Invalid request parameters
      500:
        description: Server error
    """
    try:
        data = request.get_json()
        
        # Validate required parameters
        if 'endpoint' not in data or 'response_time' not in data:
            return jsonify({'error': 'endpoint and response_time are required'}), 400
        
        # Log request
        metrics_service.log_request(
            endpoint=data['endpoint'],
            response_time=data['response_time'],
            status_code=data.get('status_code', 200),
            error=data.get('error')
        )
        
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500 