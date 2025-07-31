#!/usr/bin/env python3
"""
WSGI entry point for Student Placement Predictor
Production deployment file
"""

import os
import sys
import logging
from app import app

# Configure logging for production
if not app.debug:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('production.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

# Set environment variables for production
os.environ['FLASK_ENV'] = 'production'

if __name__ == "__main__":
    app.run() 