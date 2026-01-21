"""
Neural Network Implementation from Scratch
==========================================
Implementasi neural network tanpa sklearn untuk medical diagnosis.
Mendukung: multi-layer perceptron, berbagai activation functions, 
backpropagation, dan batch training.

Author: Medical Diagnosis System
Version: 1.0.0
"""

import math
import random
import json
from typing import List, Tuple, Dict, Callable, Optional
from copy import deepcopy


class ActivationFunctions:
    """Koleksi activation functions dan derivatifnya."""
    
    @staticmethod
    def sigmoid(x: float) -> float:
        try:
            if x < -700:
                return 0.0
            elif x > 700:
                return 1.0
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0
    
    @staticmethod
    def sigmoid_derivative(x: float) -> float:
        s = ActivationFunctions.sigmoid(x)
        return s * (1.0 - s)
    
    @staticmethod
    def relu(x: float) -> float:
        return max(0.0, x)
    
    @staticmethod
    def relu_derivative(x: float) -> float:
        return 1.0 if x > 0 else 0.0
    
    @staticmethod
    def leaky_relu(x: float, alpha: float = 0.01) -> float:
        return x if x > 0 else alpha * x
    
    @staticmethod
    def leaky_relu_derivative(x: float, alpha: float = 0.01) -> float:
        return 1.0 if x > 0 else alpha
    
    @staticmethod
    def tanh(x: float) -> float:
        try:
            return math.tanh(x)
        except OverflowError:
            return 1.0 if x > 0 else -1.0
    
    @staticmethod
    def tanh_derivative(x: float) -> float:
        t = ActivationFunctions.tanh(x)
        return 1.0 - t * t


class Layer:
    """Representasi satu layer dalam neural network."""
    
    def __init__(self, input_size: int, output_size: int, 
                 activation: str = 'relu', random_seed: Optional[int] = None):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation
        
        if random_seed is not None:
            random.seed(random_seed)
        
        # Xavier/Glorot initialization
        limit = math.sqrt(6.0 / (input_size + output_size))
        self.weights = [
            [random.uniform(-limit, limit) for _ in range(input_size)]
            for _ in range(output_size)
        ]
        self.biases = [0.0 for _ in range(output_size)]
        
        # Gradients
        self.weight_gradients = [[0.0] * input_size for _ in range(output_size)]
        self.bias_gradients = [0.0] * output_size
        
        # Cache
        self.inputs: List[float] = []
        self.z_values: List[float] = []
        self.activations: List[float] = []
        
        self._set_activation(activation)
    
    def _set_activation(self, activation: str):
        af = ActivationFunctions
        activations = {
            'sigmoid': (af.sigmoid, af.sigmoid_derivative),
            'relu': (af.relu, af.relu_derivative),
            'leaky_relu': (af.leaky_relu, af.leaky_relu_derivative),
            'tanh': (af.tanh, af.tanh_derivative)
        }
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        self.activation_func, self.activation_deriv = activations[activation]
    
    def forward(self, inputs: List[float]) -> List[float]:
        """Forward pass melalui layer."""
        self.inputs = inputs
        self.z_values = []
        self.activations = []
        
        for i in range(self.output_size):
            z = self.biases[i]
            for j in range(self.input_size):
                z += self.weights[i][j] * inputs[j]
            self.z_values.append(z)
            self.activations.append(self.activation_func(z))
        
        return self.activations
    
    def backward(self, output_gradients: List[float]) -> List[float]:
        """Backward pass untuk menghitung gradients."""
        input_gradients = [0.0] * self.input_size
        
        for i in range(self.output_size):
            delta = output_gradients[i] * self.activation_deriv(self.z_values[i])
            self.bias_gradients[i] += delta
            
            for j in range(self.input_size):
                self.weight_gradients[i][j] += delta * self.inputs[j]
                input_gradients[j] += delta * self.weights[i][j]
        
        return input_gradients
    
    def update_weights(self, learning_rate: float, batch_size: int = 1):
        """Update weights menggunakan gradients."""
        for i in range(self.output_size):
            self.biases[i] -= learning_rate * self.bias_gradients[i] / batch_size
            self.bias_gradients[i] = 0.0
            
            for j in range(self.input_size):
                self.weights[i][j] -= learning_rate * self.weight_gradients[i][j] / batch_size
                self.weight_gradients[i][j] = 0.0


class NeuralNetwork:
    """Multi-Layer Perceptron Neural Network."""
    
    def __init__(self, layer_sizes: List[int], activations: List[str] = None,
                 learning_rate: float = 0.01, random_seed: int = 42):
        """
        Initialize neural network.
        
        Parameters:
            layer_sizes: [input_size, hidden1, hidden2, ..., output_size]
            activations: Activation functions per layer (excluding input)
            learning_rate: Learning rate untuk training
            random_seed: Seed untuk reproducibility
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        
        if activations is None:
            activations = ['relu'] * (len(layer_sizes) - 2) + ['sigmoid']
        
        self.layers: List[Layer] = []
        for i in range(len(layer_sizes) - 1):
            layer = Layer(
                layer_sizes[i], 
                layer_sizes[i + 1],
                activations[i] if i < len(activations) else 'sigmoid',
                random_seed + i
            )
            self.layers.append(layer)
        
        self.training_history = {'loss': [], 'accuracy': []}
    
    def forward(self, inputs: List[float]) -> List[float]:
        """Forward pass melalui semua layers."""
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current
    
    def backward(self, target: List[float], predicted: List[float]) -> float:
        """Backward pass (backpropagation)."""
        # Calculate output layer gradients (binary cross-entropy derivative)
        output_gradients = []
        loss = 0.0
        
        for i in range(len(target)):
            p = max(min(predicted[i], 1 - 1e-15), 1e-15)
            output_gradients.append(p - target[i])
            loss -= target[i] * math.log(p) + (1 - target[i]) * math.log(1 - p)
        
        # Backpropagate through layers
        gradients = output_gradients
        for layer in reversed(self.layers):
            gradients = layer.backward(gradients)
        
        return loss
    
    def update_weights(self, batch_size: int = 1):
        """Update all layer weights."""
        for layer in self.layers:
            layer.update_weights(self.learning_rate, batch_size)
    
    def train(self, X: List[List[float]], y: List[List[float]], 
              epochs: int = 100, batch_size: int = 32, 
              verbose: bool = True, validation_split: float = 0.1) -> Dict:
        """
        Train the neural network.
        
        Parameters:
            X: Training features
            y: Training labels (one-hot encoded)
            epochs: Number of training epochs
            batch_size: Mini-batch size
            verbose: Print progress
            validation_split: Fraction for validation
            
        Returns:
            Training history
        """
        # Split data
        n_samples = len(X)
        n_val = int(n_samples * validation_split)
        
        indices = list(range(n_samples))
        random.seed(self.random_seed)
        random.shuffle(indices)
        
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        X_train = [X[i] for i in train_indices]
        y_train = [y[i] for i in train_indices]
        X_val = [X[i] for i in val_indices] if n_val > 0 else []
        y_val = [y[i] for i in val_indices] if n_val > 0 else []
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Shuffle training data
            combined = list(zip(X_train, y_train))
            random.shuffle(combined)
            X_shuffled, y_shuffled = zip(*combined)
            
            epoch_loss = 0.0
            correct = 0
            
            # Mini-batch training
            for i in range(0, len(X_shuffled), batch_size):
                batch_X = X_shuffled[i:i + batch_size]
                batch_y = y_shuffled[i:i + batch_size]
                
                batch_loss = 0.0
                for x, target in zip(batch_X, batch_y):
                    predicted = self.forward(x)
                    batch_loss += self.backward(target, predicted)
                    
                    # Calculate accuracy
                    pred_class = 1 if predicted[0] > 0.5 else 0
                    true_class = 1 if target[0] > 0.5 else 0
                    if pred_class == true_class:
                        correct += 1
                
                self.update_weights(len(batch_X))
                epoch_loss += batch_loss
            
            # Calculate metrics
            train_loss = epoch_loss / len(X_shuffled)
            train_acc = correct / len(X_shuffled)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation
            if X_val:
                val_loss, val_acc = self.evaluate(X_val, y_val)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
            
            if verbose and (epoch + 1) % 10 == 0:
                val_str = f", val_loss: {history['val_loss'][-1]:.4f}, val_acc: {history['val_acc'][-1]:.4f}" if X_val else ""
                print(f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f}, acc: {train_acc:.4f}{val_str}")
        
        self.training_history = history
        return history
    
    def evaluate(self, X: List[List[float]], y: List[List[float]]) -> Tuple[float, float]:
        """Evaluate model on data."""
        total_loss = 0.0
        correct = 0
        
        for x, target in zip(X, y):
            predicted = self.forward(x)
            
            # Calculate loss
            p = max(min(predicted[0], 1 - 1e-15), 1e-15)
            total_loss -= target[0] * math.log(p) + (1 - target[0]) * math.log(1 - p)
            
            # Calculate accuracy
            pred_class = 1 if predicted[0] > 0.5 else 0
            true_class = 1 if target[0] > 0.5 else 0
            if pred_class == true_class:
                correct += 1
        
        return total_loss / len(X), correct / len(X)
    
    def predict(self, X: List[List[float]]) -> List[float]:
        """Predict probabilities."""
        predictions = []
        for x in X:
            output = self.forward(x)
            predictions.append(output[0])
        return predictions
    
    def predict_classes(self, X: List[List[float]], threshold: float = 0.5) -> List[int]:
        """Predict class labels."""
        probabilities = self.predict(X)
        return [1 if p > threshold else 0 for p in probabilities]
    
    def save_model(self, filepath: str):
        """Save model weights to file."""
        model_data = {
            'layer_sizes': self.layer_sizes,
            'learning_rate': self.learning_rate,
            'layers': []
        }
        
        for layer in self.layers:
            layer_data = {
                'weights': layer.weights,
                'biases': layer.biases,
                'activation': layer.activation_name
            }
            model_data['layers'].append(layer_data)
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'NeuralNetwork':
        """Load model from file."""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        activations = [l['activation'] for l in model_data['layers']]
        nn = cls(model_data['layer_sizes'], activations, model_data['learning_rate'])
        
        for i, layer_data in enumerate(model_data['layers']):
            nn.layers[i].weights = layer_data['weights']
            nn.layers[i].biases = layer_data['biases']
        
        return nn


class HybridFuzzyNeuralNetwork:
    """
    Hybrid system menggabungkan Fuzzy Logic dengan Neural Network.
    Fuzzy system memberikan features tambahan untuk neural network.
    """
    
    def __init__(self, fuzzy_system, nn_layer_sizes: List[int] = None,
                 learning_rate: float = 0.01, random_seed: int = 42):
        """
        Initialize hybrid system.
        
        Parameters:
            fuzzy_system: Fuzzy expert system untuk feature extraction
            nn_layer_sizes: Layer sizes untuk neural network
            learning_rate: Learning rate
            random_seed: Random seed
        """
        self.fuzzy_system = fuzzy_system
        self.neural_network: Optional[NeuralNetwork] = None
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.nn_layer_sizes = nn_layer_sizes
        self.feature_names: List[str] = []
        self.is_trained = False
    
    def _extract_fuzzy_features(self, patient_data: Dict) -> List[float]:
        """Extract fuzzy features dari patient data."""
        # Get fuzzy memberships
        inputs = {
            'age': patient_data['age'],
            'glucose': patient_data['avg_glucose_level'],
            'bmi': patient_data['bmi'],
            'hypertension': float(patient_data['hypertension']),
            'heart_disease': float(patient_data['heart_disease'])
        }
        
        memberships = self.fuzzy_system.fuzzify_inputs(inputs)
        
        # Get fuzzy inference output
        fuzzy_output = self.fuzzy_system.infer(inputs)
        
        # Combine original features + fuzzy memberships + fuzzy output
        features = [
            patient_data['age'] / 100.0,  # Normalized
            patient_data['avg_glucose_level'] / 300.0,
            patient_data['bmi'] / 60.0,
            float(patient_data['hypertension']),
            float(patient_data['heart_disease']),
        ]
        
        # Add fuzzy memberships as features
        for var_name, term_memberships in memberships.items():
            for term_name, degree in term_memberships.items():
                features.append(degree)
        
        # Add fuzzy output
        features.append(fuzzy_output.get('stroke_risk', 50) / 100.0)
        features.append(fuzzy_output.get('severity', 5) / 10.0)
        
        return features
    
    def prepare_data(self, data: List[Dict], labels: List[int]) -> Tuple[List[List[float]], List[List[float]]]:
        """Prepare data untuk training."""
        X = []
        y = []
        
        for patient, label in zip(data, labels):
            features = self._extract_fuzzy_features(patient)
            X.append(features)
            y.append([float(label)])  # Binary classification
        
        return X, y
    
    def fit(self, data: List[Dict], labels: List[int], epochs: int = 100,
            batch_size: int = 32, verbose: bool = True) -> Dict:
        """
        Train the hybrid model.
        
        Parameters:
            data: List of patient dictionaries
            labels: List of labels (0 or 1)
            epochs: Training epochs
            batch_size: Batch size
            verbose: Print progress
            
        Returns:
            Training history
        """
        X, y = self.prepare_data(data, labels)
        
        # Initialize neural network based on feature size
        input_size = len(X[0])
        
        if self.nn_layer_sizes is None:
            self.nn_layer_sizes = [input_size, 64, 32, 16, 1]
        else:
            self.nn_layer_sizes[0] = input_size
        
        self.neural_network = NeuralNetwork(
            self.nn_layer_sizes,
            activations=['relu', 'relu', 'sigmoid'],
            learning_rate=self.learning_rate,
            random_seed=self.random_seed
        )
        
        history = self.neural_network.train(X, y, epochs, batch_size, verbose)
        self.is_trained = True
        return history
    
    def predict_proba(self, data: List[Dict]) -> List[float]:
        """Predict probabilities."""
        if not self.is_trained:
            raise RuntimeError("Model belum di-train")
        
        X = []
        for patient in data:
            features = self._extract_fuzzy_features(patient)
            X.append(features)
        
        return self.neural_network.predict(X)
    
    def predict(self, data: List[Dict], threshold: float = 0.5) -> List[int]:
        """Predict class labels."""
        probabilities = self.predict_proba(data)
        return [1 if p > threshold else 0 for p in probabilities]
    
    def predict_single(self, patient_data: Dict) -> Dict:
        """Predict untuk single patient dengan detailed output."""
        if not self.is_trained:
            raise RuntimeError("Model belum di-train")
        
        # Get fuzzy prediction
        fuzzy_result = self.fuzzy_system.predict(
            age=patient_data['age'],
            glucose=patient_data['avg_glucose_level'],
            bmi=patient_data['bmi'],
            hypertension=patient_data['hypertension'],
            heart_disease=patient_data['heart_disease']
        )
        
        # Get neural network prediction
        features = self._extract_fuzzy_features(patient_data)
        nn_probability = self.neural_network.forward(features)[0]
        
        # Combine predictions (weighted average)
        fuzzy_prob = fuzzy_result['stroke_risk_percentage'] / 100.0
        combined_prob = 0.5 * fuzzy_prob + 0.5 * nn_probability
        
        # Determine risk level
        if combined_prob < 0.2:
            risk_level = 'Sangat Rendah'
        elif combined_prob < 0.4:
            risk_level = 'Rendah'
        elif combined_prob < 0.6:
            risk_level = 'Cukup Tinggi'
        elif combined_prob < 0.8:
            risk_level = 'Tinggi'
        else:
            risk_level = 'Sangat Tinggi'
        
        return {
            'stroke_probability': round(combined_prob * 100, 2),
            'risk_level': risk_level,
            'fuzzy_risk': fuzzy_result['stroke_risk_percentage'],
            'fuzzy_severity': fuzzy_result['severity_score'],
            'fuzzy_severity_level': fuzzy_result['severity_level'],
            'nn_probability': round(nn_probability * 100, 2),
            'prediction': 1 if combined_prob > 0.5 else 0,
            'confidence': round(abs(combined_prob - 0.5) * 2 * 100, 2)
        }
    
    def save_model(self, filepath: str):
        """Save neural network weights."""
        if self.neural_network:
            self.neural_network.save_model(filepath)
    
    def load_nn_weights(self, filepath: str):
        """Load neural network weights."""
        self.neural_network = NeuralNetwork.load_model(filepath)
        self.is_trained = True


def create_hybrid_model(fuzzy_system, learning_rate: float = 0.01) -> HybridFuzzyNeuralNetwork:
    """Factory function untuk membuat hybrid model."""
    return HybridFuzzyNeuralNetwork(
        fuzzy_system=fuzzy_system,
        learning_rate=learning_rate
    )


if __name__ == "__main__":
    # Test neural network
    print("Testing Neural Network...")
    
    # Simple XOR problem
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [[0], [1], [1], [0]]
    
    nn = NeuralNetwork([2, 4, 1], ['relu', 'sigmoid'], learning_rate=0.5, random_seed=42)
    history = nn.train(X * 100, y * 100, epochs=100, batch_size=4, verbose=True, validation_split=0.0)
    
    print("\nPredictions for XOR:")
    for x in X:
        pred = nn.forward(x)
        print(f"  {x} -> {pred[0]:.4f}")
