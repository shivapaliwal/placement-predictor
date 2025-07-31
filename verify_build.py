#!/usr/bin/env python3
"""
Build Verification Script for Student Placement Predictor
Checks all components are ready for production deployment
"""

import os
import sys
import importlib
import json
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - MISSING")
        return False

def check_import(module_name, description):
    """Check if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {description}: {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {description}: {module_name} - {e}")
        return False

def check_model_files():
    """Check if all model files are present"""
    print("\n🔍 Checking Model Files...")
    model_files = [
        ('models/gradient_boosting_pipeline.pkl', 'Trained Model Pipeline'),
        ('models/scaler.pkl', 'Feature Scaler'),
        ('models/label_encoder.pkl', 'Label Encoder'),
        ('models/feature_names.pkl', 'Feature Names')
    ]
    
    all_present = True
    for filepath, description in model_files:
        if not check_file_exists(filepath, description):
            all_present = False
    
    return all_present

def check_web_files():
    """Check if all web application files are present"""
    print("\n🌐 Checking Web Application Files...")
    web_files = [
        ('app.py', 'Flask Application'),
        ('wsgi.py', 'WSGI Entry Point'),
        ('templates/index.html', 'Main Template'),
        ('templates/about.html', 'About Template'),
        ('requirements.txt', 'Dependencies'),
        ('Procfile', 'Heroku Configuration'),
        ('Dockerfile', 'Docker Configuration'),
        ('.dockerignore', 'Docker Ignore'),
        ('DEPLOYMENT.md', 'Deployment Guide')
    ]
    
    all_present = True
    for filepath, description in web_files:
        if not check_file_exists(filepath, description):
            all_present = False
    
    return all_present

def check_dependencies():
    """Check if all required dependencies can be imported"""
    print("\n📦 Checking Dependencies...")
    dependencies = [
        ('flask', 'Flask Web Framework'),
        ('pandas', 'Pandas Data Processing'),
        ('numpy', 'NumPy Numerical Computing'),
        ('sklearn', 'Scikit-learn Machine Learning'),
        ('joblib', 'Joblib Model Persistence'),
        ('matplotlib', 'Matplotlib Visualization'),
        ('seaborn', 'Seaborn Visualization')
    ]
    
    all_importable = True
    for module, description in dependencies:
        if not check_import(module, description):
            all_importable = False
    
    return all_importable

def test_model_loading():
    """Test if the model can be loaded successfully"""
    print("\n🤖 Testing Model Loading...")
    try:
        from predict import PlacementPredictor
        predictor = PlacementPredictor()
        if predictor.pipeline is not None:
            print("✅ Model loaded successfully")
            return True
        else:
            print("❌ Model failed to load")
            return False
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def test_flask_app():
    """Test if Flask app can be imported"""
    print("\n🚀 Testing Flask Application...")
    try:
        from app import app
        print("✅ Flask app imported successfully")
        
        # Test basic routes
        with app.test_client() as client:
            # Test health endpoint
            response = client.get('/api/health')
            if response.status_code == 200:
                print("✅ Health endpoint working")
            else:
                print(f"❌ Health endpoint failed: {response.status_code}")
                return False
            
            # Test home page
            response = client.get('/')
            if response.status_code == 200:
                print("✅ Home page working")
            else:
                print(f"❌ Home page failed: {response.status_code}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Flask app test failed: {e}")
        return False

def check_security_features():
    """Check if security features are implemented"""
    print("\n🔒 Checking Security Features...")
    try:
        with open('app.py', 'r') as f:
            content = f.read()
        
        security_features = [
            ('Security headers', 'add_security_headers' in content),
            ('Input validation', 'validate' in content.lower()),
            ('Error handling', 'try:' in content and 'except Exception' in content),
            ('Logging', 'logging' in content),
            ('Content Security Policy', 'Content-Security-Policy' in content)
        ]
        
        all_secure = True
        for feature, present in security_features:
            if present:
                print(f"✅ {feature}")
            else:
                print(f"❌ {feature}")
                all_secure = False
        
        return all_secure
    except Exception as e:
        print(f"❌ Security check failed: {e}")
        return False

def check_responsive_design():
    """Check if responsive design features are present"""
    print("\n📱 Checking Responsive Design...")
    try:
        with open('templates/index.html', 'r') as f:
            content = f.read()
        
        responsive_features = [
            ('Viewport meta tag', 'viewport' in content),
            ('Bootstrap CSS', 'bootstrap' in content),
            ('Media queries', '@media' in content),
            ('Responsive classes', 'col-md-' in content or 'col-lg-' in content),
            ('Mobile-friendly', 'max-width' in content)
        ]
        
        all_responsive = True
        for feature, present in responsive_features:
            if present:
                print(f"✅ {feature}")
            else:
                print(f"❌ {feature}")
                all_responsive = False
        
        return all_responsive
    except Exception as e:
        print(f"❌ Responsive design check failed: {e}")
        return False

def generate_build_report():
    """Generate a comprehensive build report"""
    print("=" * 60)
    print("🏗️  BUILD VERIFICATION REPORT")
    print("=" * 60)
    
    checks = [
        ("Model Files", check_model_files),
        ("Web Files", check_web_files),
        ("Dependencies", check_dependencies),
        ("Model Loading", test_model_loading),
        ("Flask App", test_flask_app),
        ("Security Features", check_security_features),
        ("Responsive Design", check_responsive_design)
    ]
    
    results = {}
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results[check_name] = result
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ {check_name} check failed with error: {e}")
            results[check_name] = False
            all_passed = False
    
    print("\n" + "=" * 60)
    print("📊 BUILD SUMMARY")
    print("=" * 60)
    
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name:<20} {status}")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 BUILD VERIFICATION PASSED!")
        print("✅ Your application is ready for production deployment!")
        print("\n🚀 Deployment Options:")
        print("   • Heroku: git push heroku main")
        print("   • Docker: docker build -t placement-predictor .")
        print("   • Local: py wsgi.py")
    else:
        print("⚠️  BUILD VERIFICATION FAILED!")
        print("❌ Please fix the issues above before deploying.")
    
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = generate_build_report()
    sys.exit(0 if success else 1) 