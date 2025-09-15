"""
Vercel entrypoint for FraudLens API
This file serves as the serverless function entry point for Vercel deployment
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import the FastAPI app from src/main.py
from src.main import app

# Vercel expects the app to be available as 'app'
# The app is already defined in src/main.py
