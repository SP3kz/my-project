import os
import argparse
import uvicorn

def create_directories():
    """Create necessary directories."""
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

def run_api(host="0.0.0.0", port=8000, reload=True):
    """Run the FastAPI application."""
    create_directories()
    uvicorn.run("app.main:app", host=host, port=port, reload=reload)

def run_training_example():
    """Run the training example."""
    create_directories()
    from app.examples.simple_training import main
    main()

def run_rl_example():
    """Run the reinforcement learning example."""
    create_directories()
    from app.examples.reinforcement_learning import main
    main()

def run_api_test():
    """Run the API test."""
    from app.examples.test_api import main
    main()

def main():
    """Parse arguments and run the appropriate function."""
    parser = argparse.ArgumentParser(description="ML Project Runner")
    parser.add_argument("--mode", type=str, default="api", 
                        choices=["api", "train", "rl", "test"],
                        help="Mode to run (api, train, rl, test)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to run the API on")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to run the API on")
    parser.add_argument("--no-reload", action="store_true",
                        help="Disable auto-reload for the API")
    
    args = parser.parse_args()
    
    if args.mode == "api":
        run_api(host=args.host, port=args.port, reload=not args.no_reload)
    elif args.mode == "train":
        run_training_example()
    elif args.mode == "rl":
        run_rl_example()
    elif args.mode == "test":
        run_api_test()

if __name__ == "__main__":
    main() 