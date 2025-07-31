import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class PlacementPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.feature_names = None
        
    def load_data(self):
        """Load and explore the dataset"""
        print("Loading dataset...")
        self.data = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        print(f"Data types:\n{self.data.dtypes}")
        print(f"Missing values:\n{self.data.isnull().sum()}")
        print(f"Target variable distribution:\n{self.data['Placement'].value_counts()}")
        
    def explore_data(self):
        """Explore the dataset with visualizations"""
        print("\n=== Data Exploration ===")
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Student Placement Dataset Analysis', fontsize=16)
        
        # 1. Placement distribution
        placement_counts = self.data['Placement'].value_counts()
        axes[0, 0].pie(placement_counts.values, labels=placement_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Placement Distribution')
        
        # 2. CGPA distribution by placement
        self.data.boxplot(column='CGPA', by='Placement', ax=axes[0, 1])
        axes[0, 1].set_title('CGPA Distribution by Placement')
        axes[0, 1].set_xlabel('Placement')
        
        # 3. IQ distribution by placement
        self.data.boxplot(column='IQ', by='Placement', ax=axes[0, 2])
        axes[0, 2].set_title('IQ Distribution by Placement')
        axes[0, 2].set_xlabel('Placement')
        
        # 4. Internship experience vs placement
        internship_placement = pd.crosstab(self.data['Internship_Experience'], self.data['Placement'])
        internship_placement.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Internship Experience vs Placement')
        axes[1, 0].set_xlabel('Internship Experience')
        axes[1, 0].set_ylabel('Count')
        
        # 5. Projects completed vs placement
        self.data.boxplot(column='Projects_Completed', by='Placement', ax=axes[1, 1])
        axes[1, 1].set_title('Projects Completed by Placement')
        axes[1, 1].set_xlabel('Placement')
        
        # 6. Communication skills vs placement
        self.data.boxplot(column='Communication_Skills', by='Placement', ax=axes[1, 2])
        axes[1, 2].set_title('Communication Skills by Placement')
        axes[1, 2].set_xlabel('Placement')
        
        plt.tight_layout()
        plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the plot instead of showing it
        
        # Correlation analysis
        print("\nCorrelation with Placement:")
        # Create a copy with encoded placement for correlation
        df_corr = self.data.copy()
        df_corr['Placement_Encoded'] = self.label_encoder.fit_transform(df_corr['Placement'])
        numeric_cols = df_corr.select_dtypes(include=[np.number]).columns
        correlations = df_corr[numeric_cols].corr()['Placement_Encoded'].sort_values(ascending=False)
        print(correlations)
        
    def preprocess_data(self):
        """Preprocess the data for training"""
        print("\n=== Data Preprocessing ===")
        
        # Create a copy for preprocessing
        df = self.data.copy()
        
        # Encode categorical variables
        df['Internship_Experience'] = self.label_encoder.fit_transform(df['Internship_Experience'])
        df['Placement'] = self.label_encoder.fit_transform(df['Placement'])
        
        # Drop College_ID as it's not useful for prediction
        df = df.drop('College_ID', axis=1)
        
        # Separate features and target
        X = df.drop('Placement', axis=1)
        y = df['Placement']
        
        self.feature_names = X.columns.tolist()
        print(f"Features: {self.feature_names}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Training set placement distribution: {np.bincount(self.y_train)}")
        print(f"Test set placement distribution: {np.bincount(self.y_test)}")
        
    def train_models(self):
        """Train multiple models and compare their performance"""
        print("\n=== Model Training ===")
        
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for all models
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5, scoring='accuracy')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")
            print(f"{name} - CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results
    
    def hyperparameter_tuning(self, best_model_name):
        """Perform hyperparameter tuning on the best model"""
        print(f"\n=== Hyperparameter Tuning for {best_model_name} ===")
        
        if best_model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=42)
            
        elif best_model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            model = GradientBoostingClassifier(random_state=42)
            
        elif best_model_name == 'Logistic Regression':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
            model = LogisticRegression(random_state=42, max_iter=1000)
            
        else:
            print("Hyperparameter tuning not implemented for this model")
            return None
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def evaluate_model(self, model, model_name):
        """Evaluate the final model"""
        print(f"\n=== Model Evaluation: {model_name} ===")
        
        # Make predictions
        y_pred = model.predict(self.X_test_scaled)
        y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['Not Placed', 'Placed']))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Placed', 'Placed'],
                   yticklabels=['Not Placed', 'Placed'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the plot instead of showing it
        
        # Feature importance (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_importance, x='importance', y='feature')
            plt.title(f'Feature Importance - {model_name}')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig(f'feature_importance_{model_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
            plt.close()  # Close the plot instead of showing it
            
            print("\nFeature Importance:")
            print(feature_importance)
    
    def save_model(self, model, model_name):
        """Save the trained model and preprocessing objects"""
        print(f"\n=== Saving Model: {model_name} ===")
        
        # Create a pipeline with preprocessing and model
        pipeline = Pipeline([
            ('scaler', self.scaler),
            ('classifier', model)
        ])
        
        # Save the pipeline
        model_filename = f'models/{model_name.lower().replace(" ", "_")}_pipeline.pkl'
        joblib.dump(pipeline, model_filename)
        
        # Save preprocessing objects separately
        joblib.dump(self.scaler, 'models/scaler.pkl')
        joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
        joblib.dump(self.feature_names, 'models/feature_names.pkl')
        
        print(f"Model saved as: {model_filename}")
        print("Preprocessing objects saved in models/ directory")
        
        return pipeline
    
    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        print("=== Student Placement Prediction Model Training ===")
        
        # Create models directory
        import os
        os.makedirs('models', exist_ok=True)
        
        # Load and explore data
        self.load_data()
        self.explore_data()
        
        # Preprocess data
        self.preprocess_data()
        
        # Train models
        results = self.train_models()
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_model = results[best_model_name]['model']
        
        print(f"\nBest model: {best_model_name} with accuracy: {results[best_model_name]['accuracy']:.4f}")
        
        # Hyperparameter tuning
        tuned_model = self.hyperparameter_tuning(best_model_name)
        if tuned_model is not None:
            best_model = tuned_model
        
        # Evaluate final model
        self.evaluate_model(best_model, best_model_name)
        
        # Save model
        pipeline = self.save_model(best_model, best_model_name)
        
        print("\n=== Training Complete ===")
        print("Model is ready for deployment!")
        
        return pipeline

if __name__ == "__main__":
    # Initialize and run the training pipeline
    predictor = PlacementPredictor('dataset/college_student_placement_dataset.csv')
    pipeline = predictor.run_training_pipeline() 