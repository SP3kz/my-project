import os
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

def create_app(test_config=None):
    """Create and configure the Flask application"""
    # Create Flask app
    app = Flask(__name__, instance_relative_config=True)
    
    # Configure app
    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'dev'),
        DATABASE=os.path.join(app.instance_path, 'app.sqlite'),
        JWT_SECRET_KEY=os.environ.get('JWT_SECRET_KEY', 'dev'),
        JWT_ACCESS_TOKEN_EXPIRES=int(os.environ.get('JWT_ACCESS_TOKEN_EXPIRES', 3600)),  # 1 hour
        UPLOAD_FOLDER=os.path.join(os.getcwd(), 'app', 'data', 'uploads'),
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16 MB max upload size
    )
    
    # Override config with test config if provided
    if test_config is not None:
        app.config.from_mapping(test_config)
    
    # Ensure instance folder exists
    try:
        os.makedirs(app.instance_path, exist_ok=True)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    except OSError:
        pass
    
    # Set up CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(os.getcwd(), 'app.log'))
        ]
    )
    
    # Register blueprints
    from app.routes.auth_routes import auth_bp
    from app.routes.user_routes import user_bp
    from app.routes.model_routes import model_bp
    from app.routes.dataset_routes import dataset_bp
    from app.routes.metrics_routes import metrics_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(user_bp)
    app.register_blueprint(model_bp)
    app.register_blueprint(dataset_bp)
    app.register_blueprint(metrics_bp)
    
    # Set up metrics middleware
    from app.middleware.metrics_middleware import setup_metrics_middleware
    app = setup_metrics_middleware(app)
    
    # Simple index route
    @app.route('/')
    def index():
        return {
            'message': 'Welcome to the ML Model API',
            'version': '1.0.0',
            'status': 'running'
        }
    
    return app 