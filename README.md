# ML Project with PyTorch and FastAPI

This project combines PyTorch for machine learning with FastAPI for serving predictions through a REST API.

## Features

- PyTorch-based machine learning model
- Reinforcement learning environment using OpenAI Gym
- FastAPI backend for model serving
- Easy-to-use API endpoints

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   python app/main.py
   ```

3. Access the API at http://localhost:8000

4. View API documentation at http://localhost:8000/docs

## Project Structure

- `app/`: Main application code
  - `main.py`: FastAPI application entry point
  - `models/`: PyTorch models
  - `utils/`: Utility functions
- `notebooks/`: Jupyter notebooks for experimentation
- `tests/`: Test files

## License

MIT 