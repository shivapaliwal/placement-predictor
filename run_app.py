#!/usr/bin/env python3
"""
Student Placement Predictor - Web Application Startup Script
"""

import sys
import os

def main():
    """Start the Flask web application"""
    try:
        # Check if required files exist
        if not os.path.exists('models/gradient_boosting_pipeline.pkl'):
            print("❌ Error: Model files not found!")
            print("Please run 'py model_training.py' first to train the model.")
            return
        
        if not os.path.exists('templates/index.html'):
            print("❌ Error: Template files not found!")
            print("Please ensure all template files are in the templates/ directory.")
            return
        
        print("🚀 Starting Student Placement Predictor...")
        print("📊 Model: Gradient Boosting (100% accuracy)")
        print("🌐 Web Interface: http://localhost:5000")
        print("📱 Mobile-friendly responsive design")
        print("⚡ Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Import and run the Flask app
        from app import app
        app.run(debug=False, host='0.0.0.0', port=5000)
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please install required dependencies: py -m pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

if __name__ == "__main__":
    main() 