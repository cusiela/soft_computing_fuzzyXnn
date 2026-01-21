"""
Evaluation Metrics Implementation
================================
Implementasi metrik evaluasi dari awal tanpa sklearn.
Mendukung: Accuracy, Sensitivity, Specificity, AUC-ROC, F1-Score.

Author: Medical Diagnosis System
Version: 1.0.0
"""

import math
from typing import List, Tuple, Dict


class MetricsCalculator:
    """
    Kelas untuk menghitung berbagai metrik evaluasi model.
    """
    
    @staticmethod
    def confusion_matrix(y_true: List[int], y_pred: List[int]) -> Dict[str, int]:
        """
        Hitung confusion matrix.
        
        Parameters:
            y_true: Label sebenarnya
            y_pred: Label prediksi
            
        Returns:
            Dictionary dengan TP, TN, FP, FN
        """
        tp = tn = fp = fn = 0
        
        for true, pred in zip(y_true, y_pred):
            if true == 1 and pred == 1:
                tp += 1
            elif true == 0 and pred == 0:
                tn += 1
            elif true == 0 and pred == 1:
                fp += 1
            else:  # true == 1 and pred == 0
                fn += 1
        
        return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
    
    @staticmethod
    def accuracy(y_true: List[int], y_pred: List[int]) -> float:
        """
        Hitung akurasi: (TP + TN) / (TP + TN + FP + FN)
        
        Parameters:
            y_true: Label sebenarnya
            y_pred: Label prediksi
            
        Returns:
            Accuracy score [0, 1]
        """
        cm = MetricsCalculator.confusion_matrix(y_true, y_pred)
        total = cm['TP'] + cm['TN'] + cm['FP'] + cm['FN']
        if total == 0:
            return 0.0
        return (cm['TP'] + cm['TN']) / total
    
    @staticmethod
    def sensitivity(y_true: List[int], y_pred: List[int]) -> float:
        """
        Hitung sensitivity (recall/true positive rate): TP / (TP + FN)
        
        Parameters:
            y_true: Label sebenarnya
            y_pred: Label prediksi
            
        Returns:
            Sensitivity score [0, 1]
        """
        cm = MetricsCalculator.confusion_matrix(y_true, y_pred)
        denominator = cm['TP'] + cm['FN']
        if denominator == 0:
            return 0.0
        return cm['TP'] / denominator
    
    @staticmethod
    def specificity(y_true: List[int], y_pred: List[int]) -> float:
        """
        Hitung specificity (true negative rate): TN / (TN + FP)
        
        Parameters:
            y_true: Label sebenarnya
            y_pred: Label prediksi
            
        Returns:
            Specificity score [0, 1]
        """
        cm = MetricsCalculator.confusion_matrix(y_true, y_pred)
        denominator = cm['TN'] + cm['FP']
        if denominator == 0:
            return 0.0
        return cm['TN'] / denominator
    
    @staticmethod
    def precision(y_true: List[int], y_pred: List[int]) -> float:
        """
        Hitung precision: TP / (TP + FP)
        
        Parameters:
            y_true: Label sebenarnya
            y_pred: Label prediksi
            
        Returns:
            Precision score [0, 1]
        """
        cm = MetricsCalculator.confusion_matrix(y_true, y_pred)
        denominator = cm['TP'] + cm['FP']
        if denominator == 0:
            return 0.0
        return cm['TP'] / denominator
    
    @staticmethod
    def f1_score(y_true: List[int], y_pred: List[int]) -> float:
        """
        Hitung F1-score: 2 * (precision * recall) / (precision + recall)
        
        Parameters:
            y_true: Label sebenarnya
            y_pred: Label prediksi
            
        Returns:
            F1 score [0, 1]
        """
        prec = MetricsCalculator.precision(y_true, y_pred)
        rec = MetricsCalculator.sensitivity(y_true, y_pred)
        
        if prec + rec == 0:
            return 0.0
        return 2 * (prec * rec) / (prec + rec)
    
    @staticmethod
    def roc_curve(y_true: List[int], y_scores: List[float], 
                  n_thresholds: int = 100) -> Tuple[List[float], List[float], List[float]]:
        """
        Hitung ROC curve.
        
        Parameters:
            y_true: Label sebenarnya
            y_scores: Probabilitas prediksi
            n_thresholds: Jumlah threshold points
            
        Returns:
            (fpr_list, tpr_list, thresholds)
        """
        # Generate thresholds
        thresholds = [i / n_thresholds for i in range(n_thresholds + 1)]
        thresholds = sorted(thresholds, reverse=True)
        
        fpr_list = []
        tpr_list = []
        
        for threshold in thresholds:
            y_pred = [1 if score >= threshold else 0 for score in y_scores]
            
            cm = MetricsCalculator.confusion_matrix(y_true, y_pred)
            
            # TPR (Sensitivity)
            tpr = cm['TP'] / (cm['TP'] + cm['FN']) if (cm['TP'] + cm['FN']) > 0 else 0
            
            # FPR (1 - Specificity)
            fpr = cm['FP'] / (cm['FP'] + cm['TN']) if (cm['FP'] + cm['TN']) > 0 else 0
            
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        
        return fpr_list, tpr_list, thresholds
    
    @staticmethod
    def auc_roc(y_true: List[int], y_scores: List[float]) -> float:
        """
        Hitung Area Under ROC Curve menggunakan trapezoidal rule.
        
        Parameters:
            y_true: Label sebenarnya
            y_scores: Probabilitas prediksi
            
        Returns:
            AUC score [0, 1]
        """
        fpr_list, tpr_list, _ = MetricsCalculator.roc_curve(y_true, y_scores)
        
        # Sort by FPR
        points = sorted(zip(fpr_list, tpr_list))
        
        # Calculate area using trapezoidal rule
        auc = 0.0
        for i in range(1, len(points)):
            x0, y0 = points[i - 1]
            x1, y1 = points[i]
            auc += (x1 - x0) * (y0 + y1) / 2
        
        return auc
    
    @staticmethod
    def all_metrics(y_true: List[int], y_pred: List[int], 
                    y_scores: List[float] = None) -> Dict[str, float]:
        """
        Hitung semua metrik sekaligus.
        
        Parameters:
            y_true: Label sebenarnya
            y_pred: Label prediksi
            y_scores: Probabilitas prediksi (opsional untuk AUC)
            
        Returns:
            Dictionary dengan semua metrik
        """
        cm = MetricsCalculator.confusion_matrix(y_true, y_pred)
        
        metrics = {
            'accuracy': MetricsCalculator.accuracy(y_true, y_pred),
            'sensitivity': MetricsCalculator.sensitivity(y_true, y_pred),
            'specificity': MetricsCalculator.specificity(y_true, y_pred),
            'precision': MetricsCalculator.precision(y_true, y_pred),
            'f1_score': MetricsCalculator.f1_score(y_true, y_pred),
            'confusion_matrix': cm
        }
        
        if y_scores is not None:
            metrics['auc_roc'] = MetricsCalculator.auc_roc(y_true, y_scores)
        
        return metrics
    
    @staticmethod
    def classification_report(y_true: List[int], y_pred: List[int], 
                              y_scores: List[float] = None) -> str:
        """
        Generate classification report sebagai string.
        
        Parameters:
            y_true: Label sebenarnya
            y_pred: Label prediksi
            y_scores: Probabilitas prediksi (opsional)
            
        Returns:
            Formatted report string
        """
        metrics = MetricsCalculator.all_metrics(y_true, y_pred, y_scores)
        cm = metrics['confusion_matrix']
        
        report = """
╔═══════════════════════════════════════════════════╗
║           CLASSIFICATION REPORT                    ║
╠═══════════════════════════════════════════════════╣
║                                                   ║
║  Confusion Matrix:                                ║
║  ┌─────────────┬──────────┬──────────┐           ║
║  │             │ Pred: 0  │ Pred: 1  │           ║
║  ├─────────────┼──────────┼──────────┤           ║
║  │ Actual: 0   │   {tn:^5d}  │   {fp:^5d}  │           ║
║  ├─────────────┼──────────┼──────────┤           ║
║  │ Actual: 1   │   {fn:^5d}  │   {tp:^5d}  │           ║
║  └─────────────┴──────────┴──────────┘           ║
║                                                   ║
║  Metrics:                                         ║
║  ─────────────────────────────────────           ║
║  Accuracy:    {acc:.4f}                            ║
║  Sensitivity: {sens:.4f} (Recall/TPR)              ║
║  Specificity: {spec:.4f} (TNR)                     ║
║  Precision:   {prec:.4f}                            ║
║  F1-Score:    {f1:.4f}                             ║""".format(
            tn=cm['TN'], fp=cm['FP'], fn=cm['FN'], tp=cm['TP'],
            acc=metrics['accuracy'],
            sens=metrics['sensitivity'],
            spec=metrics['specificity'],
            prec=metrics['precision'],
            f1=metrics['f1_score']
        )
        
        if 'auc_roc' in metrics:
            report += f"""
║  AUC-ROC:     {metrics['auc_roc']:.4f}                             ║"""
        
        report += """
║                                                   ║
╚═══════════════════════════════════════════════════╝
"""
        return report


def calculate_all_metrics(y_true: List[int], y_pred: List[int], 
                          y_scores: List[float] = None) -> Dict[str, float]:
    """Helper function untuk menghitung semua metrik."""
    return MetricsCalculator.all_metrics(y_true, y_pred, y_scores)


def print_classification_report(y_true: List[int], y_pred: List[int], 
                                y_scores: List[float] = None):
    """Helper function untuk print classification report."""
    print(MetricsCalculator.classification_report(y_true, y_pred, y_scores))


if __name__ == "__main__":
    # Test metrics
    y_true = [0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
    y_pred = [0, 1, 1, 1, 0, 0, 0, 1, 1, 0]
    y_scores = [0.1, 0.6, 0.8, 0.9, 0.2, 0.3, 0.15, 0.85, 0.75, 0.25]
    
    print_classification_report(y_true, y_pred, y_scores)
