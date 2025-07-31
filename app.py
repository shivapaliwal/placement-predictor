from flask import Flask, render_template, request, jsonify
from predict import PlacementPredictor
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Security headers
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; font-src 'self' https://cdnjs.cloudflare.com; img-src 'self' data:;"
    return response

# Initialize the predictor
try:
    predictor = PlacementPredictor()
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    predictor = None

@app.route('/')
def home():
    """Home page with the prediction form"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering home page: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for making predictions"""
    try:
        # Check if model is loaded
        if predictor is None or predictor.pipeline is None:
            return jsonify({'error': 'Model not available'}), 503
        
        # Get form data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['iq', 'prev_sem_result', 'cgpa', 'academic_performance', 
                          'internship_experience', 'extra_curricular_score', 
                          'communication_skills', 'projects_completed']
        
        missing_fields = []
        for field in required_fields:
            if field not in data or data[field] == '':
                missing_fields.append(field)
        
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400
        
        # Validate data types and ranges
        try:
            student_data = {
                'IQ': int(data['iq']),
                'Prev_Sem_Result': float(data['prev_sem_result']),
                'CGPA': float(data['cgpa']),
                'Academic_Performance': int(data['academic_performance']),
                'Internship_Experience': data['internship_experience'],
                'Extra_Curricular_Score': int(data['extra_curricular_score']),
                'Communication_Skills': int(data['communication_skills']),
                'Projects_Completed': int(data['projects_completed'])
            }
        except (ValueError, TypeError) as e:
            return jsonify({'error': f'Invalid data format: {str(e)}'}), 400
        
        # Validate ranges
        if not (70 <= student_data['IQ'] <= 150):
            return jsonify({'error': 'IQ must be between 70 and 150'}), 400
        if not (0 <= student_data['Prev_Sem_Result'] <= 10):
            return jsonify({'error': 'Previous semester result must be between 0 and 10'}), 400
        if not (0 <= student_data['CGPA'] <= 10):
            return jsonify({'error': 'CGPA must be between 0 and 10'}), 400
        if not (1 <= student_data['Academic_Performance'] <= 10):
            return jsonify({'error': 'Academic performance must be between 1 and 10'}), 400
        if student_data['Internship_Experience'] not in ['Yes', 'No']:
            return jsonify({'error': 'Internship experience must be Yes or No'}), 400
        if not (0 <= student_data['Extra_Curricular_Score'] <= 10):
            return jsonify({'error': 'Extra-curricular score must be between 0 and 10'}), 400
        if not (1 <= student_data['Communication_Skills'] <= 10):
            return jsonify({'error': 'Communication skills must be between 1 and 10'}), 400
        if not (0 <= student_data['Projects_Completed'] <= 20):
            return jsonify({'error': 'Projects completed must be between 0 and 20'}), 400
        
        # Make prediction
        result = predictor.predict_placement(student_data)
        
        if 'error' in result:
            logger.error(f"Prediction error: {result['error']}")
            return jsonify({'error': result['error']}), 500
        
        # Log successful prediction
        logger.info(f"Successful prediction: {result['placement_prediction']} for student with CGPA: {student_data['CGPA']}")
        
        return jsonify({
            'success': True,
            'prediction': result['placement_prediction'],
            'placement_probability': round(result['placement_probability'] * 100, 2),
            'not_placed_probability': round(result['not_placed_probability'] * 100, 2),
            'confidence': result['confidence'],
            'student_data': student_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/about')
def about():
    """About page with model information"""
    try:
        return render_template('about.html')
    except Exception as e:
        logger.error(f"Error rendering about page: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            'status': 'healthy',
            'model_loaded': predictor is not None and predictor.pipeline is not None,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/api/model-info')
def model_info():
    """Get model information"""
    try:
        if predictor is None or predictor.pipeline is None:
            return jsonify({'error': 'Model not available'}), 503
        
        return jsonify({
            'model_type': 'Gradient Boosting Classifier',
            'accuracy': '100%',
            'roc_auc': '1.0000',
            'features': predictor.feature_names if hasattr(predictor, 'feature_names') else [],
            'feature_importance': {
                'CGPA': 37.1,
                'Communication_Skills': 22.9,
                'IQ': 21.4,
                'Projects_Completed': 18.4,
                'Prev_Sem_Result': 0.2,
                'Academic_Performance': 0.0,
                'Internship_Experience': 0.0,
                'Extra_Curricular_Score': 0.0
            }
        })
    except Exception as e:
        logger.error(f"Model info error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Page not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Production settings
    app.run(
        debug=False,  # Disable debug mode for production
        host='0.0.0.0', 
        port=int(os.environ.get('PORT', 5000)),
        threaded=True
    ) 