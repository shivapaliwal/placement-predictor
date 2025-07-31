import joblib
import pandas as pd
import numpy as np

class PlacementPredictor:
    def __init__(self, model_path='models/gradient_boosting_pipeline.pkl'):
        """Initialize the predictor with a trained model"""
        try:
            self.pipeline = joblib.load(model_path)
            self.scaler = joblib.load('models/scaler.pkl')
            self.label_encoder = joblib.load('models/label_encoder.pkl')
            self.feature_names = joblib.load('models/feature_names.pkl')
            print("Model loaded successfully!")
        except FileNotFoundError:
            print("Model files not found. Please run model_training.py first.")
            self.pipeline = None
    
    def predict_placement(self, student_data):
        """
        Predict placement for a single student
        
        Parameters:
        student_data (dict): Dictionary containing student features
            - IQ: int/float
            - Prev_Sem_Result: float
            - CGPA: float
            - Academic_Performance: int (1-10)
            - Internship_Experience: str ('Yes' or 'No')
            - Extra_Curricular_Score: int (0-10)
            - Communication_Skills: int (0-10)
            - Projects_Completed: int
        
        Returns:
        dict: Prediction results with placement probability and class
        """
        if self.pipeline is None:
            return {"error": "Model not loaded"}
        
        try:
            # Create DataFrame from student data
            df = pd.DataFrame([student_data])
            
            # Encode categorical variables
            df['Internship_Experience'] = self.label_encoder.transform(df['Internship_Experience'])
            
            # Ensure correct column order
            df = df[self.feature_names]
            
            # Make prediction
            prediction = self.pipeline.predict(df)[0]
            probability = self.pipeline.predict_proba(df)[0]
            
            # Decode prediction
            placement_status = self.label_encoder.inverse_transform([prediction])[0]
            
            return {
                "placement_prediction": placement_status,
                "placement_probability": float(probability[1]),  # Probability of being placed
                "not_placed_probability": float(probability[0]),  # Probability of not being placed
                "confidence": "High" if max(probability) > 0.8 else "Medium" if max(probability) > 0.6 else "Low"
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def predict_batch(self, students_data):
        """
        Predict placement for multiple students
        
        Parameters:
        students_data (list): List of dictionaries containing student features
        
        Returns:
        list: List of prediction results
        """
        results = []
        for student_data in students_data:
            result = self.predict_placement(student_data)
            results.append(result)
        return results

def main():
    """Example usage of the prediction model"""
    # Initialize predictor
    predictor = PlacementPredictor()
    
    if predictor.pipeline is None:
        return
    
    # Example student data
    example_student = {
        'IQ': 110,
        'Prev_Sem_Result': 8.5,
        'CGPA': 8.2,
        'Academic_Performance': 8,
        'Internship_Experience': 'Yes',
        'Extra_Curricular_Score': 7,
        'Communication_Skills': 8,
        'Projects_Completed': 3
    }
    
    print("Example Student Data:")
    for key, value in example_student.items():
        print(f"  {key}: {value}")
    
    print("\nMaking prediction...")
    result = predictor.predict_placement(example_student)
    
    if "error" not in result:
        print(f"\nPrediction Results:")
        print(f"  Placement Prediction: {result['placement_prediction']}")
        print(f"  Placement Probability: {result['placement_probability']:.2%}")
        print(f"  Not Placed Probability: {result['not_placed_probability']:.2%}")
        print(f"  Confidence: {result['confidence']}")
    else:
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    main() 