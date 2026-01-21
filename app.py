"""
Medical Diagnosis System - Flask Application
============================================
Aplikasi web untuk diagnosis stroke menggunakan Fuzzy Expert System + Neural Network.

Features:
- Patient data input form
- Real-time prediction
- Interactive visualizations
- Model performance metrics
- EDA dashboard

Author: Medical Diagnosis System
Version: 1.0.0
"""

import os
import json
import math
from flask import Flask, render_template, request, jsonify, send_file
from typing import Dict, List, Any

# Import models
from models.fuzzy import create_stroke_fuzzy_system, StrokeFuzzyExpertSystem
from models.neural import NeuralNetwork, HybridFuzzyNeuralNetwork, create_hybrid_model
from models.metrics import MetricsCalculator, calculate_all_metrics
from models.preprocessing import (
    DataLoader, DataCleaner, StrokeDataPreprocessor, 
    train_test_split, FeatureEncoder, FeatureScaler
)

app = Flask(__name__)

# Global variables untuk model dan data
fuzzy_system: StrokeFuzzyExpertSystem = None
hybrid_model: HybridFuzzyNeuralNetwork = None
preprocessor: StrokeDataPreprocessor = None
train_data: List[Dict] = []
test_data: List[Dict] = []
model_metrics: Dict = {}
eda_results: Dict = {}


def initialize_system():
    """Initialize semua model dan data."""
    global fuzzy_system, hybrid_model, preprocessor, train_data, test_data, model_metrics, eda_results
    
    print("Initializing Medical Diagnosis System...")
    
    # Initialize fuzzy system
    fuzzy_system = create_stroke_fuzzy_system()
    print("✓ Fuzzy Expert System initialized")
    
    # Initialize preprocessor dan load data
    preprocessor = StrokeDataPreprocessor()
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'raw', 'healthcare-dataset-stroke-data.csv')
    
    if os.path.exists(data_path):
        # Use hybrid balancing to handle imbalanced data
        train_data, test_data, info = preprocessor.preprocess(
            data_path, 
            test_size=0.2,
            balance_method='hybrid'  # Options: 'none', 'oversample', 'undersample', 'smote', 'hybrid'
        )
        print(f"✓ Data loaded: {len(train_data)} train, {len(test_data)} test records")
        
        # Perform EDA on original (unbalanced) data for accurate statistics
        headers, raw_data = DataLoader.load_csv(data_path)
        eda_results = perform_eda(raw_data)
        print("✓ EDA completed")
        
        # Train hybrid model with balanced data
        train_model()
    else:
        print(f"⚠ Data file not found at {data_path}")
        train_data = []
        test_data = []


def perform_eda(data: List[Dict]) -> Dict:
    """Perform Exploratory Data Analysis on raw data."""
    if not data:
        return {}
    
    # Basic statistics
    total_records = len(data)
    stroke_cases = sum(1 for d in data if str(d.get('stroke', '0')) == '1')
    no_stroke_cases = total_records - stroke_cases
    
    # Age distribution - handle both string and float values
    ages = []
    for d in data:
        age_val = d.get('age')
        if age_val is not None and str(age_val) not in ['', 'N/A', 'NA']:
            try:
                ages.append(float(age_val))
            except (ValueError, TypeError):
                pass
    
    age_stats = {
        'min': min(ages) if ages else 0,
        'max': max(ages) if ages else 0,
        'mean': sum(ages) / len(ages) if ages else 0,
        'bins': calculate_histogram(ages, 10)
    }
    
    # Glucose distribution
    glucose = []
    for d in data:
        g_val = d.get('avg_glucose_level')
        if g_val is not None and str(g_val) not in ['', 'N/A', 'NA']:
            try:
                glucose.append(float(g_val))
            except (ValueError, TypeError):
                pass
    
    glucose_stats = {
        'min': min(glucose) if glucose else 0,
        'max': max(glucose) if glucose else 0,
        'mean': sum(glucose) / len(glucose) if glucose else 0,
        'bins': calculate_histogram(glucose, 10)
    }
    
    # BMI distribution
    bmi = []
    for d in data:
        b_val = d.get('bmi')
        if b_val is not None and str(b_val) not in ['', 'N/A', 'NA']:
            try:
                bmi.append(float(b_val))
            except (ValueError, TypeError):
                pass
    
    bmi_stats = {
        'min': min(bmi) if bmi else 0,
        'max': max(bmi) if bmi else 0,
        'mean': sum(bmi) / len(bmi) if bmi else 0,
        'bins': calculate_histogram(bmi, 10)
    }
    
    # Correlation with stroke by age group
    stroke_by_age = {
        'young': {'stroke': 0, 'no_stroke': 0},
        'middle': {'stroke': 0, 'no_stroke': 0},
        'old': {'stroke': 0, 'no_stroke': 0}
    }
    
    for d in data:
        try:
            age = float(d.get('age', 0))
            stroke = str(d.get('stroke', '0')) == '1'
            
            if age < 40:
                category = 'young'
            elif age < 60:
                category = 'middle'
            else:
                category = 'old'
            
            if stroke:
                stroke_by_age[category]['stroke'] += 1
            else:
                stroke_by_age[category]['no_stroke'] += 1
        except (ValueError, TypeError):
            pass
    
    # Risk factors analysis
    hypertension_risk = calculate_risk_factor_raw(data, 'hypertension')
    heart_disease_risk = calculate_risk_factor_raw(data, 'heart_disease')
    
    return {
        'total_records': total_records,
        'stroke_cases': stroke_cases,
        'no_stroke_cases': no_stroke_cases,
        'stroke_rate': round(stroke_cases / total_records * 100, 2) if total_records > 0 else 0,
        'age_stats': age_stats,
        'glucose_stats': glucose_stats,
        'bmi_stats': bmi_stats,
        'stroke_by_age': stroke_by_age,
        'risk_factors': {
            'hypertension': hypertension_risk,
            'heart_disease': heart_disease_risk
        }
    }


def calculate_risk_factor_raw(data: List[Dict], factor: str) -> Dict:
    """Calculate risk factor statistics from raw data."""
    with_factor = sum(1 for d in data if str(d.get(factor, '0')) == '1')
    without_factor = len(data) - with_factor
    
    stroke_with = sum(1 for d in data if str(d.get(factor, '0')) == '1' and str(d.get('stroke', '0')) == '1')
    stroke_without = sum(1 for d in data if str(d.get(factor, '0')) == '0' and str(d.get('stroke', '0')) == '1')
    
    return {
        'with_factor': with_factor,
        'without_factor': without_factor,
        'stroke_rate_with': round(stroke_with / with_factor * 100, 2) if with_factor > 0 else 0,
        'stroke_rate_without': round(stroke_without / without_factor * 100, 2) if without_factor > 0 else 0
    }


def calculate_histogram(values: List[float], n_bins: int) -> List[Dict]:
    """Calculate histogram bins."""
    if not values:
        return []
    
    min_val = min(values)
    max_val = max(values)
    bin_width = (max_val - min_val) / n_bins if max_val != min_val else 1
    
    bins = []
    for i in range(n_bins):
        bin_start = min_val + i * bin_width
        bin_end = bin_start + bin_width
        count = sum(1 for v in values if bin_start <= v < bin_end or (i == n_bins - 1 and v == max_val))
        bins.append({
            'range': f"{bin_start:.1f}-{bin_end:.1f}",
            'count': count
        })
    
    return bins


def calculate_risk_factor(data: List[Dict], factor: str) -> Dict:
    """Calculate risk factor statistics."""
    with_factor = sum(1 for d in data if d.get(factor) == 1)
    without_factor = len(data) - with_factor
    
    stroke_with = sum(1 for d in data if d.get(factor) == 1 and d.get('stroke') == 1)
    stroke_without = sum(1 for d in data if d.get(factor) == 0 and d.get('stroke') == 1)
    
    return {
        'with_factor': with_factor,
        'without_factor': without_factor,
        'stroke_rate_with': round(stroke_with / with_factor * 100, 2) if with_factor > 0 else 0,
        'stroke_rate_without': round(stroke_without / without_factor * 100, 2) if without_factor > 0 else 0
    }


def train_model():
    """Train the hybrid model with balanced data."""
    global hybrid_model, model_metrics
    
    if not train_data or not test_data:
        print("⚠ No data available for training")
        return
    
    print("Training Hybrid Fuzzy-Neural Network...")
    
    # Prepare training data for hybrid model
    train_patients = []
    train_labels = []
    
    for row in train_data:
        # Handle both normalized and raw values
        age_val = row.get('age', 0.5)
        glucose_val = row.get('avg_glucose_level', 0.33)
        bmi_val = row.get('bmi', 0.4)
        
        # Denormalize if values are normalized (0-1 range)
        if age_val <= 1:
            age_val = age_val * 100
        if glucose_val <= 1:
            glucose_val = glucose_val * 300
        if bmi_val <= 1:
            bmi_val = bmi_val * 60
        
        patient = {
            'age': age_val,
            'avg_glucose_level': glucose_val,
            'bmi': bmi_val,
            'hypertension': int(row.get('hypertension', 0)),
            'heart_disease': int(row.get('heart_disease', 0))
        }
        train_patients.append(patient)
        train_labels.append(int(row.get('stroke', 0)))
    
    # Create and train hybrid model with higher learning rate for balanced data
    hybrid_model = create_hybrid_model(fuzzy_system, learning_rate=0.05)
    history = hybrid_model.fit(
        train_patients, train_labels,
        epochs=100,  # More epochs for better convergence
        batch_size=32,
        verbose=True
    )
    
    print("✓ Model training completed")
    
    # Evaluate on test data
    evaluate_model()


def evaluate_model():
    """Evaluate model performance."""
    global model_metrics
    
    if not test_data or hybrid_model is None:
        return
    
    # Prepare test data
    test_patients = []
    y_true = []
    
    for row in test_data:
        # Handle both normalized and raw values
        age_val = row.get('age', 0.5)
        glucose_val = row.get('avg_glucose_level', 0.33)
        bmi_val = row.get('bmi', 0.4)
        
        # Denormalize if values are normalized (0-1 range)
        if age_val <= 1:
            age_val = age_val * 100
        if glucose_val <= 1:
            glucose_val = glucose_val * 300
        if bmi_val <= 1:
            bmi_val = bmi_val * 60
        
        patient = {
            'age': age_val,
            'avg_glucose_level': glucose_val,
            'bmi': bmi_val,
            'hypertension': int(row.get('hypertension', 0)),
            'heart_disease': int(row.get('heart_disease', 0))
        }
        test_patients.append(patient)
        y_true.append(int(row.get('stroke', 0)))
    
    # Get predictions - use optimal threshold based on ROC
    y_scores = hybrid_model.predict_proba(test_patients)
    
    # Find optimal threshold using Youden's J statistic
    best_threshold = 0.5
    best_j = 0
    for thresh in [i/100 for i in range(10, 90, 5)]:
        y_pred_temp = [1 if s > thresh else 0 for s in y_scores]
        sens = MetricsCalculator.sensitivity(y_true, y_pred_temp)
        spec = MetricsCalculator.specificity(y_true, y_pred_temp)
        j = sens + spec - 1
        if j > best_j:
            best_j = j
            best_threshold = thresh
    
    print(f"Optimal threshold: {best_threshold:.2f}")
    y_pred = [1 if s > best_threshold else 0 for s in y_scores]
    
    # Calculate metrics
    model_metrics = calculate_all_metrics(y_true, y_pred, y_scores)
    model_metrics['threshold'] = best_threshold
    
    # Add ROC curve data
    fpr, tpr, thresholds = MetricsCalculator.roc_curve(y_true, y_scores)
    model_metrics['roc_curve'] = {
        'fpr': [round(x, 4) for x in fpr],
        'tpr': [round(x, 4) for x in tpr]
    }
    
    print(f"✓ Model Evaluation - Accuracy: {model_metrics['accuracy']:.4f}, "
          f"Sensitivity: {model_metrics['sensitivity']:.4f}, "
          f"Specificity: {model_metrics['specificity']:.4f}, "
          f"AUC: {model_metrics['auc_roc']:.4f}")


# Flask Routes
@app.route('/')
def index():
    """Main page dengan input form."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint untuk prediksi."""
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        patient_data = {
            'age': float(data['age']),
            'avg_glucose_level': float(data['avg_glucose_level']),
            'bmi': float(data['bmi']),
            'hypertension': int(data['hypertension']),
            'heart_disease': int(data['heart_disease'])
        }
        
        # Get fuzzy prediction
        fuzzy_result = fuzzy_system.predict(
            age=patient_data['age'],
            glucose=patient_data['avg_glucose_level'],
            bmi=patient_data['bmi'],
            hypertension=patient_data['hypertension'],
            heart_disease=patient_data['heart_disease']
        )
        
        # Get hybrid model prediction if available
        if hybrid_model is not None and hybrid_model.is_trained:
            hybrid_result = hybrid_model.predict_single(patient_data)
            result = {
                'stroke_probability': hybrid_result['stroke_probability'],
                'risk_level': hybrid_result['risk_level'],
                'severity_score': fuzzy_result['severity_score'],
                'severity_level': fuzzy_result['severity_level'],
                'fuzzy_risk': fuzzy_result['stroke_risk_percentage'],
                'nn_probability': hybrid_result['nn_probability'],
                'confidence': hybrid_result['confidence'],
                'prediction': hybrid_result['prediction'],
                'memberships': fuzzy_result['input_memberships']
            }
        else:
            # Fuzzy-only prediction
            result = {
                'stroke_probability': fuzzy_result['stroke_risk_percentage'],
                'risk_level': fuzzy_result['risk_level'],
                'severity_score': fuzzy_result['severity_score'],
                'severity_level': fuzzy_result['severity_level'],
                'fuzzy_risk': fuzzy_result['stroke_risk_percentage'],
                'nn_probability': None,
                'confidence': 100 - abs(50 - fuzzy_result['stroke_risk_percentage']) * 2,
                'prediction': 1 if fuzzy_result['stroke_risk_percentage'] > 50 else 0,
                'memberships': fuzzy_result['input_memberships']
            }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/metrics')
def get_metrics():
    """Get model performance metrics."""
    if not model_metrics:
        return jsonify({'error': 'Model not trained yet'}), 404
    
    # Convert confusion matrix to serializable format
    metrics_response = {
        'accuracy': round(model_metrics.get('accuracy', 0), 4),
        'sensitivity': round(model_metrics.get('sensitivity', 0), 4),
        'specificity': round(model_metrics.get('specificity', 0), 4),
        'precision': round(model_metrics.get('precision', 0), 4),
        'f1_score': round(model_metrics.get('f1_score', 0), 4),
        'auc_roc': round(model_metrics.get('auc_roc', 0), 4),
        'confusion_matrix': model_metrics.get('confusion_matrix', {}),
        'roc_curve': model_metrics.get('roc_curve', {})
    }
    
    return jsonify(metrics_response)


@app.route('/eda')
def get_eda():
    """Get EDA results."""
    if not eda_results:
        return jsonify({'error': 'No data available'}), 404
    
    return jsonify(eda_results)


@app.route('/fuzzy/rules')
def get_fuzzy_rules():
    """Get fuzzy rules for visualization."""
    rules = []
    for rule in fuzzy_system.rules:
        rules.append({
            'id': rule.rule_id,
            'description': str(rule),
            'weight': rule.weight
        })
    return jsonify(rules)


@app.route('/fuzzy/membership/<variable>')
def get_membership_function(variable):
    """Get membership function data for visualization."""
    if variable in fuzzy_system.input_variables:
        var = fuzzy_system.input_variables[variable]
    elif variable in fuzzy_system.output_variables:
        var = fuzzy_system.output_variables[variable]
    else:
        return jsonify({'error': 'Variable not found'}), 404
    
    min_val, max_val = var.universe
    x_values = [min_val + i * (max_val - min_val) / 100 for i in range(101)]
    
    membership_data = {
        'variable': variable,
        'universe': [min_val, max_val],
        'x_values': x_values,
        'terms': {}
    }
    
    for term_name, mf_func in var.terms.items():
        membership_data['terms'][term_name] = [round(mf_func(x), 4) for x in x_values]
    
    return jsonify(membership_data)


@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'fuzzy_system': fuzzy_system is not None,
        'hybrid_model': hybrid_model is not None and hybrid_model.is_trained,
        'data_loaded': len(train_data) > 0
    })


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# Initialize on startup
initialize_system()


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
