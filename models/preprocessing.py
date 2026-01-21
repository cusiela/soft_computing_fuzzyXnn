"""
Data Preprocessing Module
=========================
Modul untuk data cleaning, preprocessing, dan train-test split.
Implementasi dari scratch tanpa sklearn.

Author: Medical Diagnosis System
Version: 1.0.0
"""

import csv
import math
import random
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter


class DataLoader:
    """Kelas untuk memuat data dari file CSV."""
    
    @staticmethod
    def load_csv(filepath: str, has_header: bool = True) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Load data dari file CSV.
        
        Parameters:
            filepath: Path ke file CSV
            has_header: Apakah file memiliki header
            
        Returns:
            (headers, data) dimana data adalah list of dictionaries
        """
        data = []
        headers = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            
            if has_header:
                headers = next(reader)
            
            for row in reader:
                if headers:
                    row_dict = {}
                    for i, header in enumerate(headers):
                        if i < len(row):
                            row_dict[header] = row[i]
                    data.append(row_dict)
                else:
                    data.append(row)
        
        return headers, data
    
    @staticmethod
    def save_csv(filepath: str, data: List[Dict], headers: List[str] = None):
        """Save data ke file CSV."""
        if not data:
            return
        
        if headers is None:
            headers = list(data[0].keys())
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)


class DataCleaner:
    """Kelas untuk membersihkan data."""
    
    @staticmethod
    def convert_types(data: List[Dict], type_mapping: Dict[str, type]) -> List[Dict]:
        """
        Konversi tipe data kolom.
        
        Parameters:
            data: List of dictionaries
            type_mapping: {column_name: target_type}
            
        Returns:
            Data dengan tipe yang sudah dikonversi
        """
        cleaned_data = []
        
        for row in data:
            new_row = row.copy()
            for col, target_type in type_mapping.items():
                if col in new_row:
                    try:
                        if target_type == float:
                            new_row[col] = float(new_row[col]) if new_row[col] not in ['', 'N/A', 'NA', None] else None
                        elif target_type == int:
                            new_row[col] = int(float(new_row[col])) if new_row[col] not in ['', 'N/A', 'NA', None] else None
                        elif target_type == str:
                            new_row[col] = str(new_row[col])
                    except (ValueError, TypeError):
                        new_row[col] = None
            cleaned_data.append(new_row)
        
        return cleaned_data
    
    @staticmethod
    def handle_missing_values(data: List[Dict], strategy: Dict[str, str], 
                              fill_values: Dict[str, Any] = None) -> List[Dict]:
        """
        Handle missing values dengan berbagai strategi.
        
        Parameters:
            data: List of dictionaries
            strategy: {column_name: 'mean'|'median'|'mode'|'drop'|'fill'}
            fill_values: {column_name: value} untuk strategi 'fill'
            
        Returns:
            Data yang sudah di-handle missing values-nya
        """
        if not data:
            return data
        
        fill_values = fill_values or {}
        
        # Calculate statistics untuk setiap kolom
        stats = {}
        for col, strat in strategy.items():
            values = [row[col] for row in data if row.get(col) is not None and row[col] != '']
            
            if strat == 'mean' and values:
                try:
                    numeric_values = [float(v) for v in values]
                    stats[col] = sum(numeric_values) / len(numeric_values)
                except (ValueError, TypeError):
                    stats[col] = None
                    
            elif strat == 'median' and values:
                try:
                    numeric_values = sorted([float(v) for v in values])
                    n = len(numeric_values)
                    mid = n // 2
                    stats[col] = numeric_values[mid] if n % 2 else (numeric_values[mid-1] + numeric_values[mid]) / 2
                except (ValueError, TypeError):
                    stats[col] = None
                    
            elif strat == 'mode' and values:
                counter = Counter(values)
                stats[col] = counter.most_common(1)[0][0]
                
            elif strat == 'fill':
                stats[col] = fill_values.get(col)
        
        # Apply filling
        cleaned_data = []
        for row in data:
            new_row = row.copy()
            should_drop = False
            
            for col, strat in strategy.items():
                if new_row.get(col) is None or new_row.get(col) == '' or new_row.get(col) == 'N/A':
                    if strat == 'drop':
                        should_drop = True
                        break
                    elif col in stats and stats[col] is not None:
                        new_row[col] = stats[col]
            
            if not should_drop:
                cleaned_data.append(new_row)
        
        return cleaned_data
    
    @staticmethod
    def remove_duplicates(data: List[Dict], subset: List[str] = None) -> List[Dict]:
        """Remove duplicate rows."""
        seen = set()
        unique_data = []
        
        for row in data:
            if subset:
                key = tuple(row.get(col) for col in subset)
            else:
                key = tuple(sorted(row.items()))
            
            if key not in seen:
                seen.add(key)
                unique_data.append(row)
        
        return unique_data


class FeatureEncoder:
    """Kelas untuk encoding fitur kategorikal."""
    
    def __init__(self):
        self.encodings: Dict[str, Dict[str, int]] = {}
        self.inverse_encodings: Dict[str, Dict[int, str]] = {}
    
    def fit(self, data: List[Dict], columns: List[str]):
        """
        Fit encoder pada data.
        
        Parameters:
            data: List of dictionaries
            columns: Kolom yang akan di-encode
        """
        for col in columns:
            unique_values = sorted(set(row.get(col) for row in data if row.get(col) is not None))
            self.encodings[col] = {val: idx for idx, val in enumerate(unique_values)}
            self.inverse_encodings[col] = {idx: val for val, idx in self.encodings[col].items()}
    
    def transform(self, data: List[Dict], columns: List[str]) -> List[Dict]:
        """
        Transform data menggunakan fitted encoder.
        
        Parameters:
            data: List of dictionaries
            columns: Kolom yang akan di-transform
            
        Returns:
            Data yang sudah di-encode
        """
        transformed_data = []
        
        for row in data:
            new_row = row.copy()
            for col in columns:
                if col in self.encodings and col in new_row:
                    value = new_row[col]
                    new_row[col] = self.encodings[col].get(value, -1)
            transformed_data.append(new_row)
        
        return transformed_data
    
    def fit_transform(self, data: List[Dict], columns: List[str]) -> List[Dict]:
        """Fit dan transform dalam satu langkah."""
        self.fit(data, columns)
        return self.transform(data, columns)
    
    def inverse_transform(self, data: List[Dict], columns: List[str]) -> List[Dict]:
        """Convert encoded values kembali ke original."""
        inverse_data = []
        
        for row in data:
            new_row = row.copy()
            for col in columns:
                if col in self.inverse_encodings and col in new_row:
                    encoded_value = new_row[col]
                    new_row[col] = self.inverse_encodings[col].get(encoded_value, None)
            inverse_data.append(new_row)
        
        return inverse_data


class FeatureScaler:
    """Kelas untuk scaling fitur numerik."""
    
    def __init__(self, method: str = 'minmax'):
        """
        Initialize scaler.
        
        Parameters:
            method: 'minmax' atau 'standard'
        """
        self.method = method
        self.stats: Dict[str, Dict[str, float]] = {}
    
    def fit(self, data: List[Dict], columns: List[str]):
        """
        Fit scaler pada data.
        
        Parameters:
            data: List of dictionaries
            columns: Kolom yang akan di-scale
        """
        for col in columns:
            values = [float(row[col]) for row in data 
                      if row.get(col) is not None and str(row[col]) not in ['', 'N/A']]
            
            if not values:
                continue
            
            if self.method == 'minmax':
                self.stats[col] = {
                    'min': min(values),
                    'max': max(values)
                }
            elif self.method == 'standard':
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                self.stats[col] = {
                    'mean': mean,
                    'std': math.sqrt(variance) if variance > 0 else 1.0
                }
    
    def transform(self, data: List[Dict], columns: List[str]) -> List[Dict]:
        """
        Transform data menggunakan fitted scaler.
        
        Parameters:
            data: List of dictionaries
            columns: Kolom yang akan di-transform
            
        Returns:
            Data yang sudah di-scale
        """
        transformed_data = []
        
        for row in data:
            new_row = row.copy()
            for col in columns:
                if col in self.stats and col in new_row:
                    try:
                        value = float(new_row[col])
                        
                        if self.method == 'minmax':
                            min_val = self.stats[col]['min']
                            max_val = self.stats[col]['max']
                            if max_val != min_val:
                                new_row[col] = (value - min_val) / (max_val - min_val)
                            else:
                                new_row[col] = 0.5
                                
                        elif self.method == 'standard':
                            mean = self.stats[col]['mean']
                            std = self.stats[col]['std']
                            new_row[col] = (value - mean) / std
                            
                    except (ValueError, TypeError):
                        pass
            transformed_data.append(new_row)
        
        return transformed_data
    
    def fit_transform(self, data: List[Dict], columns: List[str]) -> List[Dict]:
        """Fit dan transform dalam satu langkah."""
        self.fit(data, columns)
        return self.transform(data, columns)
    
    def inverse_transform(self, data: List[Dict], columns: List[str]) -> List[Dict]:
        """Convert scaled values kembali ke original."""
        inverse_data = []
        
        for row in data:
            new_row = row.copy()
            for col in columns:
                if col in self.stats and col in new_row:
                    try:
                        value = float(new_row[col])
                        
                        if self.method == 'minmax':
                            min_val = self.stats[col]['min']
                            max_val = self.stats[col]['max']
                            new_row[col] = value * (max_val - min_val) + min_val
                            
                        elif self.method == 'standard':
                            mean = self.stats[col]['mean']
                            std = self.stats[col]['std']
                            new_row[col] = value * std + mean
                            
                    except (ValueError, TypeError):
                        pass
            inverse_data.append(new_row)
        
        return inverse_data


class DataBalancer:
    """
    Kelas untuk menangani imbalanced dataset.
    Mendukung: SMOTE-like oversampling, Random Undersampling, dan Combined approach.
    """
    
    @staticmethod
    def random_oversample(data: List[Dict], target_column: str, 
                          random_seed: int = 42) -> List[Dict]:
        """
        Random Oversampling: Duplikasi sampel dari kelas minoritas.
        
        Parameters:
            data: List of dictionaries
            target_column: Nama kolom target
            random_seed: Seed untuk reproducibility
            
        Returns:
            Balanced data dengan oversampling
        """
        random.seed(random_seed)
        
        # Pisahkan berdasarkan kelas
        class_0 = [d for d in data if d.get(target_column) == 0]
        class_1 = [d for d in data if d.get(target_column) == 1]
        
        majority = class_0 if len(class_0) > len(class_1) else class_1
        minority = class_1 if len(class_0) > len(class_1) else class_0
        
        print(f"  Before oversampling: Majority={len(majority)}, Minority={len(minority)}")
        
        # Oversample minority class
        n_to_add = len(majority) - len(minority)
        oversampled = minority.copy()
        
        for _ in range(n_to_add):
            # Random sample with replacement
            sample = random.choice(minority).copy()
            oversampled.append(sample)
        
        balanced_data = majority + oversampled
        random.shuffle(balanced_data)
        
        print(f"  After oversampling: Total={len(balanced_data)}")
        
        return balanced_data
    
    @staticmethod
    def random_undersample(data: List[Dict], target_column: str,
                           random_seed: int = 42) -> List[Dict]:
        """
        Random Undersampling: Kurangi sampel dari kelas mayoritas.
        
        Parameters:
            data: List of dictionaries
            target_column: Nama kolom target
            random_seed: Seed untuk reproducibility
            
        Returns:
            Balanced data dengan undersampling
        """
        random.seed(random_seed)
        
        # Pisahkan berdasarkan kelas
        class_0 = [d for d in data if d.get(target_column) == 0]
        class_1 = [d for d in data if d.get(target_column) == 1]
        
        majority = class_0 if len(class_0) > len(class_1) else class_1
        minority = class_1 if len(class_0) > len(class_1) else class_0
        
        print(f"  Before undersampling: Majority={len(majority)}, Minority={len(minority)}")
        
        # Undersample majority class
        undersampled_majority = random.sample(majority, len(minority))
        
        balanced_data = undersampled_majority + minority
        random.shuffle(balanced_data)
        
        print(f"  After undersampling: Total={len(balanced_data)}")
        
        return balanced_data
    
    @staticmethod
    def smote_oversample(data: List[Dict], target_column: str,
                         feature_columns: List[str], k_neighbors: int = 5,
                         random_seed: int = 42) -> List[Dict]:
        """
        SMOTE (Synthetic Minority Over-sampling Technique).
        Membuat sampel sintetis berdasarkan interpolasi k-nearest neighbors.
        
        Parameters:
            data: List of dictionaries
            target_column: Nama kolom target
            feature_columns: Kolom fitur untuk interpolasi
            k_neighbors: Jumlah neighbors untuk SMOTE
            random_seed: Seed untuk reproducibility
            
        Returns:
            Balanced data dengan synthetic samples
        """
        random.seed(random_seed)
        
        # Pisahkan berdasarkan kelas
        class_0 = [d for d in data if d.get(target_column) == 0]
        class_1 = [d for d in data if d.get(target_column) == 1]
        
        majority = class_0 if len(class_0) > len(class_1) else class_1
        minority = class_1 if len(class_0) > len(class_1) else class_0
        minority_label = 1 if len(class_0) > len(class_1) else 0
        
        print(f"  Before SMOTE: Majority={len(majority)}, Minority={len(minority)}")
        
        n_synthetic = len(majority) - len(minority)
        synthetic_samples = []
        
        # Extract features untuk distance calculation
        def get_features(sample):
            return [float(sample.get(col, 0) or 0) for col in feature_columns]
        
        def euclidean_distance(a, b):
            return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
        
        minority_features = [get_features(s) for s in minority]
        
        # Generate synthetic samples
        for _ in range(n_synthetic):
            # Pick random minority sample
            idx = random.randint(0, len(minority) - 1)
            sample = minority[idx]
            sample_features = minority_features[idx]
            
            # Find k nearest neighbors
            distances = []
            for j, feat in enumerate(minority_features):
                if j != idx:
                    dist = euclidean_distance(sample_features, feat)
                    distances.append((j, dist))
            
            distances.sort(key=lambda x: x[1])
            k = min(k_neighbors, len(distances))
            
            if k == 0:
                # Jika tidak ada neighbor, duplikasi saja
                synthetic_samples.append(sample.copy())
                continue
            
            # Pick random neighbor from k nearest
            neighbor_idx = distances[random.randint(0, k - 1)][0]
            neighbor = minority[neighbor_idx]
            neighbor_features = minority_features[neighbor_idx]
            
            # Interpolate
            alpha = random.random()
            synthetic = sample.copy()
            
            for i, col in enumerate(feature_columns):
                if isinstance(sample.get(col), (int, float)) and sample.get(col) is not None:
                    new_val = sample_features[i] + alpha * (neighbor_features[i] - sample_features[i])
                    synthetic[col] = new_val
            
            synthetic[target_column] = minority_label
            synthetic_samples.append(synthetic)
        
        balanced_data = majority + minority + synthetic_samples
        random.shuffle(balanced_data)
        
        print(f"  After SMOTE: Total={len(balanced_data)}, Synthetic={len(synthetic_samples)}")
        
        return balanced_data
    
    @staticmethod
    def hybrid_sampling(data: List[Dict], target_column: str,
                        feature_columns: List[str] = None,
                        oversample_ratio: float = 0.5,
                        random_seed: int = 42) -> List[Dict]:
        """
        Hybrid approach: Kombinasi undersampling mayoritas dan oversampling minoritas.
        Lebih balanced approach yang menghindari extreme data loss atau overfitting.
        
        Parameters:
            data: List of dictionaries
            target_column: Nama kolom target
            feature_columns: Kolom fitur untuk SMOTE (optional)
            oversample_ratio: Target ratio minority/majority (0-1)
            random_seed: Seed untuk reproducibility
            
        Returns:
            Balanced data dengan hybrid sampling
        """
        random.seed(random_seed)
        
        # Pisahkan berdasarkan kelas
        class_0 = [d for d in data if d.get(target_column) == 0]
        class_1 = [d for d in data if d.get(target_column) == 1]
        
        majority = class_0 if len(class_0) > len(class_1) else class_1
        minority = class_1 if len(class_0) > len(class_1) else class_0
        
        print(f"  Before hybrid sampling: Majority={len(majority)}, Minority={len(minority)}")
        
        # Target size: geometric mean atau ratio-based
        target_size = int(math.sqrt(len(majority) * len(minority)) * (1 + oversample_ratio))
        target_size = max(target_size, len(minority) * 2)  # At least 2x minority
        target_size = min(target_size, len(majority))  # Not more than majority
        
        # Undersample majority
        undersampled_majority = random.sample(majority, target_size)
        
        # Oversample minority to match
        oversampled_minority = minority.copy()
        while len(oversampled_minority) < target_size:
            sample = random.choice(minority).copy()
            # Add small noise to numeric features
            for key, value in sample.items():
                if isinstance(value, float) and key != target_column:
                    noise = random.gauss(0, 0.01 * abs(value) if value != 0 else 0.01)
                    sample[key] = value + noise
            oversampled_minority.append(sample)
        
        balanced_data = undersampled_majority + oversampled_minority
        random.shuffle(balanced_data)
        
        final_majority = sum(1 for d in balanced_data if d.get(target_column) == (0 if len(class_0) > len(class_1) else 1))
        final_minority = len(balanced_data) - final_majority
        
        print(f"  After hybrid: Total={len(balanced_data)}, Ratio={final_minority/final_majority:.2f}")
        
        return balanced_data


def train_test_split(data: List[Dict], test_size: float = 0.2, 
                     random_seed: int = 42, stratify_column: str = None) -> Tuple[List[Dict], List[Dict]]:
    """
    Split data menjadi training dan testing sets.
    
    Parameters:
        data: List of dictionaries
        test_size: Proporsi data untuk testing (0-1)
        random_seed: Seed untuk reproducibility
        stratify_column: Kolom untuk stratified split
        
    Returns:
        (train_data, test_data)
    """
    random.seed(random_seed)
    
    if stratify_column:
        # Stratified split
        groups: Dict[Any, List[Dict]] = {}
        for row in data:
            key = row.get(stratify_column)
            if key not in groups:
                groups[key] = []
            groups[key].append(row)
        
        train_data = []
        test_data = []
        
        for key, group_data in groups.items():
            random.shuffle(group_data)
            n_test = max(1, int(len(group_data) * test_size))
            test_data.extend(group_data[:n_test])
            train_data.extend(group_data[n_test:])
    else:
        # Random split
        data_copy = data.copy()
        random.shuffle(data_copy)
        
        n_test = int(len(data_copy) * test_size)
        test_data = data_copy[:n_test]
        train_data = data_copy[n_test:]
    
    return train_data, test_data


class StrokeDataPreprocessor:
    """
    Preprocessor khusus untuk dataset stroke.
    Menggabungkan semua langkah preprocessing termasuk data balancing.
    """
    
    def __init__(self):
        self.encoder = FeatureEncoder()
        self.scaler = FeatureScaler(method='minmax')
        self.categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        self.numerical_columns = ['age', 'avg_glucose_level', 'bmi']
        self.type_mapping = {
            'id': int,
            'age': float,
            'hypertension': int,
            'heart_disease': int,
            'avg_glucose_level': float,
            'bmi': float,
            'stroke': int
        }
    
    def preprocess(self, filepath: str, test_size: float = 0.2,
                   balance_method: str = 'smote') -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Preprocess stroke data secara lengkap dengan data balancing.
        
        Parameters:
            filepath: Path ke file CSV
            test_size: Proporsi data testing
            balance_method: 'none', 'oversample', 'undersample', 'smote', 'hybrid'
            
        Returns:
            (train_data, test_data, preprocessing_info)
        """
        # Load data
        headers, raw_data = DataLoader.load_csv(filepath)
        print(f"Loaded {len(raw_data)} records")
        
        # Convert types
        data = DataCleaner.convert_types(raw_data, self.type_mapping)
        print(f"After type conversion: {len(data)} records")
        
        # Handle missing values
        missing_strategy = {
            'bmi': 'median',
            'smoking_status': 'mode',
            'age': 'mean',
            'avg_glucose_level': 'mean'
        }
        data = DataCleaner.handle_missing_values(data, missing_strategy)
        print(f"After handling missing values: {len(data)} records")
        
        # Remove duplicates
        data = DataCleaner.remove_duplicates(data, subset=['id'])
        print(f"After removing duplicates: {len(data)} records")
        
        # Check class distribution before balancing
        stroke_count = sum(1 for d in data if d.get('stroke') == 1)
        no_stroke_count = len(data) - stroke_count
        print(f"Class distribution: Stroke={stroke_count}, No Stroke={no_stroke_count}, Ratio={stroke_count/no_stroke_count:.4f}")
        
        # Encode categorical columns
        data = self.encoder.fit_transform(data, self.categorical_columns)
        
        # Split data BEFORE balancing (important to avoid data leakage)
        train_data, test_data = train_test_split(
            data, test_size=test_size, stratify_column='stroke'
        )
        print(f"Train: {len(train_data)}, Test: {len(test_data)}")
        
        # Apply balancing ONLY to training data
        if balance_method != 'none':
            print(f"Applying {balance_method} balancing to training data...")
            
            if balance_method == 'oversample':
                train_data = DataBalancer.random_oversample(train_data, 'stroke')
            elif balance_method == 'undersample':
                train_data = DataBalancer.random_undersample(train_data, 'stroke')
            elif balance_method == 'smote':
                train_data = DataBalancer.smote_oversample(
                    train_data, 'stroke', 
                    self.numerical_columns + ['hypertension', 'heart_disease'],
                    k_neighbors=3
                )
            elif balance_method == 'hybrid':
                train_data = DataBalancer.hybrid_sampling(
                    train_data, 'stroke',
                    self.numerical_columns,
                    oversample_ratio=0.5
                )
        
        # Check class distribution after balancing
        train_stroke = sum(1 for d in train_data if d.get('stroke') == 1)
        train_no_stroke = len(train_data) - train_stroke
        print(f"After balancing - Train: Stroke={train_stroke}, No Stroke={train_no_stroke}")
        
        # Scale numerical columns (fit on train, transform both)
        self.scaler.fit(train_data, self.numerical_columns)
        train_data_scaled = self.scaler.transform(train_data, self.numerical_columns)
        test_data_scaled = self.scaler.transform(test_data, self.numerical_columns)
        
        # Preprocessing info
        info = {
            'total_records': len(raw_data),
            'train_records': len(train_data),
            'test_records': len(test_data),
            'balance_method': balance_method,
            'features': self.categorical_columns + self.numerical_columns,
            'target': 'stroke',
            'encoder_mappings': self.encoder.encodings,
            'scaler_stats': self.scaler.stats
        }
        
        return train_data_scaled, test_data_scaled, info
    
    def prepare_for_model(self, data: List[Dict]) -> Tuple[List[List[float]], List[int]]:
        """
        Prepare data untuk model training/prediction.
        
        Parameters:
            data: Preprocessed data
            
        Returns:
            (X, y) features dan labels
        """
        feature_cols = ['age', 'hypertension', 'heart_disease', 
                        'avg_glucose_level', 'bmi', 'gender',
                        'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        
        X = []
        y = []
        
        for row in data:
            features = []
            for col in feature_cols:
                val = row.get(col, 0)
                features.append(float(val) if val is not None else 0.0)
            X.append(features)
            y.append(int(row.get('stroke', 0)))
        
        return X, y
    
    def get_patient_dict(self, row: Dict) -> Dict:
        """Convert preprocessed row ke patient dictionary untuk model."""
        return {
            'age': float(row.get('age', 50)) * 100,  # Denormalize
            'avg_glucose_level': float(row.get('avg_glucose_level', 100)) * 300,
            'bmi': float(row.get('bmi', 25)) * 60,
            'hypertension': int(row.get('hypertension', 0)),
            'heart_disease': int(row.get('heart_disease', 0))
        }


if __name__ == "__main__":
    # Test preprocessing
    import os
    
    # Test dengan sample data
    sample_data = [
        {'id': '1', 'gender': 'Male', 'age': '67', 'hypertension': '0', 
         'heart_disease': '1', 'ever_married': 'Yes', 'work_type': 'Private',
         'Residence_type': 'Urban', 'avg_glucose_level': '228.69', 
         'bmi': '36.6', 'smoking_status': 'formerly smoked', 'stroke': '1'},
        {'id': '2', 'gender': 'Female', 'age': '61', 'hypertension': '0',
         'heart_disease': '0', 'ever_married': 'Yes', 'work_type': 'Self-employed',
         'Residence_type': 'Rural', 'avg_glucose_level': '202.21',
         'bmi': 'N/A', 'smoking_status': 'never smoked', 'stroke': '1'}
    ]
    
    print("Testing DataCleaner...")
    
    # Convert types
    type_mapping = {'age': float, 'hypertension': int, 'bmi': float, 'stroke': int}
    cleaned = DataCleaner.convert_types(sample_data, type_mapping)
    print(f"After type conversion: {cleaned[1]['bmi']}")  # Should be None
    
    # Handle missing
    cleaned = DataCleaner.handle_missing_values(cleaned, {'bmi': 'mean'})
    print(f"After handling missing: {cleaned[1]['bmi']}")  # Should be 36.6 (mean)
    
    print("\nPreprocessing test completed!")
