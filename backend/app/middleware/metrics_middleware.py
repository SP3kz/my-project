import time
import logging
from functools import wraps
from flask import request, g
import requests
import json

logger = logging.getLogger(__name__)

def metrics_middleware():
    """
    Middleware to track API request metrics
    
    This middleware measures the response time of each request and logs it
    for performance monitoring.
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Record start time
            start_time = time.time()
            
            # Set start time in Flask's g object for access in other parts of the request
            g.start_time = start_time
            
            # Process the request
            try:
                response = f(*args, **kwargs)
                status_code = response.status_code
                error = None
            except Exception as e:
                # Log the error
                logger.exception("Error processing request")
                status_code = 500
                error = str(e)
                # Re-raise the exception
                raise
            finally:
                # Calculate response time
                end_time = time.time()
                response_time = end_time - start_time
                
                # Log metrics asynchronously
                try:
                    # Get endpoint from request
                    endpoint = request.path
                    
                    # Log metrics in a non-blocking way
                    _log_metrics_async(endpoint, response_time, status_code, error)
                except Exception as e:
                    # Don't let metrics logging failure affect the response
                    logger.error(f"Error logging metrics: {str(e)}")
            
            return response
        return decorated_function
    return decorator

def _log_metrics_async(endpoint, response_time, status_code, error=None):
    """
    Log metrics asynchronously to avoid blocking the response
    
    In a production environment, this would use a proper async task queue
    like Celery, but for simplicity we're using a separate thread.
    """
    try:
        # Prepare log data
        log_data = {
            'endpoint': endpoint,
            'response_time': response_time,
            'status_code': status_code
        }
        
        if error:
            log_data['error'] = error
        
        # In a real application, you might use a message queue or background worker
        # For this example, we'll make a non-blocking HTTP request to our metrics endpoint
        
        # Use a separate thread to avoid blocking
        from threading import Thread
        
        def send_log_request():
            try:
                # Make request to metrics logging endpoint
                # Use internal URL to avoid going through the load balancer
                requests.post(
                    'http://localhost:5000/api/metrics/log',
                    json=log_data,
                    timeout=1  # Short timeout to avoid hanging
                )
            except Exception as e:
                # Log error but don't propagate
                logger.error(f"Failed to send metrics log: {str(e)}")
        
        # Start thread
        Thread(target=send_log_request).start()
    except Exception as e:
        logger.error(f"Error in async metrics logging: {str(e)}")

def setup_metrics_middleware(app):
    """
    Set up metrics middleware for all routes
    
    This function adds a before_request and after_request handler to the Flask app
    to track metrics for all requests.
    """
    @app.before_request
    def before_request():
        # Record start time
        g.start_time = time.time()
    
    @app.after_request
    def after_request(response):
        # Skip metrics endpoint to avoid infinite loop
        if request.path == '/api/metrics/log':
            return response
        
        try:
            # Calculate response time
            response_time = time.time() - g.start_time
            
            # Get endpoint
            endpoint = request.path
            
            # Get status code
            status_code = response.status_code
            
            # Log metrics asynchronously
            _log_metrics_async(endpoint, response_time, status_code)
        except Exception as e:
            # Don't let metrics logging failure affect the response
            logger.error(f"Error in after_request metrics logging: {str(e)}")
        
        return response
    
    return app 