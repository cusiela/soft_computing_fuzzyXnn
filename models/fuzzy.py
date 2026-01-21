"""
Fuzzy Logic Expert System Implementation
========================================
Implementasi sistem fuzzy dari awal tanpa library sklearn.
Digunakan untuk medical diagnosis dengan membership functions dan fuzzy rules.

Author: Medical Diagnosis System
Version: 1.0.0
"""

import math
from typing import Dict, List, Tuple, Callable, Optional


class MembershipFunction:
    """
    Kelas untuk mendefinisikan berbagai jenis membership function.
    Mendukung: triangular, trapezoidal, gaussian, dan sigmoid.
    """
    
    @staticmethod
    def triangular(x: float, a: float, b: float, c: float) -> float:
        """
        Triangular membership function.
        
        Parameters:
            x: Input value
            a: Left foot of triangle
            b: Peak of triangle
            c: Right foot of triangle
            
        Returns:
            Membership degree [0, 1]
        """
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a) if b != a else 1.0
        else:  # b < x < c
            return (c - x) / (c - b) if c != b else 1.0
    
    @staticmethod
    def trapezoidal(x: float, a: float, b: float, c: float, d: float) -> float:
        """
        Trapezoidal membership function.
        
        Parameters:
            x: Input value
            a: Left foot
            b: Left shoulder
            c: Right shoulder
            d: Right foot
            
        Returns:
            Membership degree [0, 1]
        """
        if x <= a or x >= d:
            return 0.0
        elif a < x < b:
            return (x - a) / (b - a) if b != a else 1.0
        elif b <= x <= c:
            return 1.0
        else:  # c < x < d
            return (d - x) / (d - c) if d != c else 1.0
    
    @staticmethod
    def gaussian(x: float, mean: float, sigma: float) -> float:
        """
        Gaussian membership function.
        
        Parameters:
            x: Input value
            mean: Center of gaussian
            sigma: Standard deviation
            
        Returns:
            Membership degree [0, 1]
        """
        if sigma == 0:
            return 1.0 if x == mean else 0.0
        return math.exp(-0.5 * ((x - mean) / sigma) ** 2)
    
    @staticmethod
    def sigmoid(x: float, a: float, c: float) -> float:
        """
        Sigmoid membership function.
        
        Parameters:
            x: Input value
            a: Slope parameter
            c: Inflection point
            
        Returns:
            Membership degree [0, 1]
        """
        try:
            return 1.0 / (1.0 + math.exp(-a * (x - c)))
        except OverflowError:
            return 0.0 if -a * (x - c) > 0 else 1.0


class FuzzyVariable:
    """
    Representasi variabel fuzzy dengan multiple linguistic terms.
    """
    
    def __init__(self, name: str, universe: Tuple[float, float]):
        """
        Initialize fuzzy variable.
        
        Parameters:
            name: Nama variabel
            universe: Range nilai (min, max)
        """
        self.name = name
        self.universe = universe
        self.terms: Dict[str, Callable[[float], float]] = {}
    
    def add_term(self, term_name: str, mf_func: Callable[[float], float]) -> 'FuzzyVariable':
        """
        Tambahkan linguistic term ke variabel.
        
        Parameters:
            term_name: Nama term (e.g., 'low', 'medium', 'high')
            mf_func: Membership function
            
        Returns:
            Self for chaining
        """
        self.terms[term_name] = mf_func
        return self
    
    def fuzzify(self, value: float) -> Dict[str, float]:
        """
        Konversi crisp value ke fuzzy membership degrees.
        
        Parameters:
            value: Crisp input value
            
        Returns:
            Dictionary of {term_name: membership_degree}
        """
        return {term: mf(value) for term, mf in self.terms.items()}
    
    def get_membership(self, value: float, term: str) -> float:
        """
        Dapatkan membership degree untuk term tertentu.
        
        Parameters:
            value: Crisp input value
            term: Nama linguistic term
            
        Returns:
            Membership degree
        """
        if term not in self.terms:
            raise ValueError(f"Term '{term}' not found in variable '{self.name}'")
        return self.terms[term](value)


class FuzzyRule:
    """
    Representasi aturan fuzzy IF-THEN.
    """
    
    def __init__(self, rule_id: int, antecedents: List[Tuple[str, str]], 
                 consequent: Tuple[str, str], weight: float = 1.0):
        """
        Initialize fuzzy rule.
        
        Parameters:
            rule_id: ID unik untuk rule
            antecedents: List of (variable_name, term_name) untuk kondisi IF
            consequent: (variable_name, term_name) untuk hasil THEN
            weight: Bobot rule [0, 1]
        """
        self.rule_id = rule_id
        self.antecedents = antecedents
        self.consequent = consequent
        self.weight = weight
    
    def evaluate(self, memberships: Dict[str, Dict[str, float]], 
                 operator: str = 'AND') -> float:
        """
        Evaluasi rule dengan membership values yang diberikan.
        
        Parameters:
            memberships: {variable_name: {term_name: degree}}
            operator: 'AND' (min) atau 'OR' (max)
            
        Returns:
            Firing strength dari rule
        """
        degrees = []
        for var_name, term_name in self.antecedents:
            if var_name in memberships and term_name in memberships[var_name]:
                degrees.append(memberships[var_name][term_name])
            else:
                degrees.append(0.0)
        
        if not degrees:
            return 0.0
        
        if operator == 'AND':
            strength = min(degrees)
        elif operator == 'OR':
            strength = max(degrees)
        else:
            raise ValueError(f"Unknown operator: {operator}")
        
        return strength * self.weight
    
    def __repr__(self) -> str:
        ant_str = " AND ".join([f"{var} IS {term}" for var, term in self.antecedents])
        cons_str = f"{self.consequent[0]} IS {self.consequent[1]}"
        return f"Rule {self.rule_id}: IF {ant_str} THEN {cons_str} (weight={self.weight})"


class FuzzyExpertSystem:
    """
    Sistem pakar fuzzy lengkap untuk medical diagnosis.
    Mengimplementasikan: fuzzification, rule evaluation, aggregation, dan defuzzification.
    """
    
    def __init__(self, name: str = "Medical Fuzzy Expert System"):
        """
        Initialize fuzzy expert system.
        
        Parameters:
            name: Nama sistem
        """
        self.name = name
        self.input_variables: Dict[str, FuzzyVariable] = {}
        self.output_variables: Dict[str, FuzzyVariable] = {}
        self.rules: List[FuzzyRule] = []
    
    def add_input_variable(self, variable: FuzzyVariable) -> 'FuzzyExpertSystem':
        """Tambahkan variabel input."""
        self.input_variables[variable.name] = variable
        return self
    
    def add_output_variable(self, variable: FuzzyVariable) -> 'FuzzyExpertSystem':
        """Tambahkan variabel output."""
        self.output_variables[variable.name] = variable
        return self
    
    def add_rule(self, rule: FuzzyRule) -> 'FuzzyExpertSystem':
        """Tambahkan aturan fuzzy."""
        self.rules.append(rule)
        return self
    
    def fuzzify_inputs(self, crisp_inputs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Fuzzifikasi semua input values.
        
        Parameters:
            crisp_inputs: {variable_name: crisp_value}
            
        Returns:
            {variable_name: {term_name: membership_degree}}
        """
        memberships = {}
        for var_name, value in crisp_inputs.items():
            if var_name in self.input_variables:
                memberships[var_name] = self.input_variables[var_name].fuzzify(value)
        return memberships
    
    def evaluate_rules(self, memberships: Dict[str, Dict[str, float]]) -> Dict[str, List[Tuple[str, float]]]:
        """
        Evaluasi semua rules dan return firing strengths.
        
        Parameters:
            memberships: Fuzzified input values
            
        Returns:
            {output_var: [(term, strength), ...]}
        """
        output_activations: Dict[str, List[Tuple[str, float]]] = {
            var: [] for var in self.output_variables
        }
        
        for rule in self.rules:
            strength = rule.evaluate(memberships)
            if strength > 0:
                out_var, out_term = rule.consequent
                if out_var in output_activations:
                    output_activations[out_var].append((out_term, strength))
        
        return output_activations
    
    def aggregate(self, activations: List[Tuple[str, float]], 
                  output_var: FuzzyVariable, resolution: int = 100) -> Tuple[List[float], List[float]]:
        """
        Agregasi output membership functions menggunakan metode maximum.
        
        Parameters:
            activations: List of (term, strength)
            output_var: Output fuzzy variable
            resolution: Jumlah titik untuk diskritisasi
            
        Returns:
            (x_values, aggregated_membership_values)
        """
        min_val, max_val = output_var.universe
        x_values = [min_val + i * (max_val - min_val) / resolution for i in range(resolution + 1)]
        aggregated = []
        
        for x in x_values:
            max_membership = 0.0
            for term, strength in activations:
                if term in output_var.terms:
                    # Clipping method: min of membership and strength
                    membership = min(output_var.terms[term](x), strength)
                    max_membership = max(max_membership, membership)
            aggregated.append(max_membership)
        
        return x_values, aggregated
    
    def defuzzify_centroid(self, x_values: List[float], memberships: List[float]) -> float:
        """
        Defuzzifikasi menggunakan metode centroid (center of gravity).
        
        Parameters:
            x_values: Domain values
            memberships: Aggregated membership values
            
        Returns:
            Crisp output value
        """
        numerator = sum(x * m for x, m in zip(x_values, memberships))
        denominator = sum(memberships)
        
        if denominator == 0:
            return (x_values[0] + x_values[-1]) / 2  # Return center if no activation
        
        return numerator / denominator
    
    def defuzzify_bisector(self, x_values: List[float], memberships: List[float]) -> float:
        """
        Defuzzifikasi menggunakan metode bisector.
        
        Parameters:
            x_values: Domain values
            memberships: Aggregated membership values
            
        Returns:
            Crisp output value
        """
        total_area = sum(memberships)
        if total_area == 0:
            return (x_values[0] + x_values[-1]) / 2
        
        half_area = total_area / 2
        cumulative = 0.0
        
        for i, (x, m) in enumerate(zip(x_values, memberships)):
            cumulative += m
            if cumulative >= half_area:
                return x
        
        return x_values[-1]
    
    def defuzzify_mom(self, x_values: List[float], memberships: List[float]) -> float:
        """
        Defuzzifikasi menggunakan metode Mean of Maximum.
        
        Parameters:
            x_values: Domain values
            memberships: Aggregated membership values
            
        Returns:
            Crisp output value
        """
        if not memberships or max(memberships) == 0:
            return (x_values[0] + x_values[-1]) / 2
        
        max_membership = max(memberships)
        max_indices = [i for i, m in enumerate(memberships) if m == max_membership]
        
        return sum(x_values[i] for i in max_indices) / len(max_indices)
    
    def infer(self, crisp_inputs: Dict[str, float], 
              defuzz_method: str = 'centroid') -> Dict[str, float]:
        """
        Lakukan inferensi fuzzy lengkap.
        
        Parameters:
            crisp_inputs: {variable_name: crisp_value}
            defuzz_method: 'centroid', 'bisector', atau 'mom'
            
        Returns:
            {output_variable_name: crisp_output_value}
        """
        # Step 1: Fuzzification
        memberships = self.fuzzify_inputs(crisp_inputs)
        
        # Step 2: Rule Evaluation
        activations = self.evaluate_rules(memberships)
        
        # Step 3: Aggregation and Defuzzification
        outputs = {}
        for out_var_name, out_var in self.output_variables.items():
            if activations[out_var_name]:
                x_values, aggregated = self.aggregate(activations[out_var_name], out_var)
                
                if defuzz_method == 'centroid':
                    outputs[out_var_name] = self.defuzzify_centroid(x_values, aggregated)
                elif defuzz_method == 'bisector':
                    outputs[out_var_name] = self.defuzzify_bisector(x_values, aggregated)
                elif defuzz_method == 'mom':
                    outputs[out_var_name] = self.defuzzify_mom(x_values, aggregated)
                else:
                    raise ValueError(f"Unknown defuzzification method: {defuzz_method}")
            else:
                # Default output if no rules fired
                outputs[out_var_name] = (out_var.universe[0] + out_var.universe[1]) / 2
        
        return outputs
    
    def get_rule_activations(self, crisp_inputs: Dict[str, float]) -> List[Tuple[FuzzyRule, float]]:
        """
        Dapatkan aktivasi semua rules untuk interpretasi.
        
        Parameters:
            crisp_inputs: Input values
            
        Returns:
            List of (rule, firing_strength)
        """
        memberships = self.fuzzify_inputs(crisp_inputs)
        activations = []
        
        for rule in self.rules:
            strength = rule.evaluate(memberships)
            activations.append((rule, strength))
        
        return activations


class StrokeFuzzyExpertSystem(FuzzyExpertSystem):
    """
    Sistem pakar fuzzy khusus untuk prediksi stroke.
    Menggunakan variabel: age, glucose, bmi, hypertension, heart_disease.
    """
    
    def __init__(self):
        super().__init__("Stroke Risk Fuzzy Expert System")
        self._setup_variables()
        self._setup_rules()
    
    def _setup_variables(self):
        """Setup semua variabel fuzzy untuk stroke prediction."""
        mf = MembershipFunction
        
        # Age variable (0-100)
        age = FuzzyVariable('age', (0, 100))
        age.add_term('young', lambda x: mf.trapezoidal(x, 0, 0, 25, 40))
        age.add_term('middle', lambda x: mf.triangular(x, 30, 50, 70))
        age.add_term('old', lambda x: mf.trapezoidal(x, 60, 75, 100, 100))
        self.add_input_variable(age)
        
        # Glucose level variable (50-300)
        glucose = FuzzyVariable('glucose', (50, 300))
        glucose.add_term('low', lambda x: mf.trapezoidal(x, 50, 50, 70, 100))
        glucose.add_term('normal', lambda x: mf.triangular(x, 70, 100, 140))
        glucose.add_term('high', lambda x: mf.trapezoidal(x, 126, 180, 300, 300))
        self.add_input_variable(glucose)
        
        # BMI variable (10-60)
        bmi = FuzzyVariable('bmi', (10, 60))
        bmi.add_term('underweight', lambda x: mf.trapezoidal(x, 10, 10, 16, 18.5))
        bmi.add_term('normal', lambda x: mf.triangular(x, 17, 22, 25))
        bmi.add_term('overweight', lambda x: mf.triangular(x, 24, 27.5, 30))
        bmi.add_term('obese', lambda x: mf.trapezoidal(x, 29, 35, 60, 60))
        self.add_input_variable(bmi)
        
        # Hypertension (0-1, treated as fuzzy)
        hypertension = FuzzyVariable('hypertension', (0, 1))
        hypertension.add_term('no', lambda x: mf.triangular(x, -0.5, 0, 0.5))
        hypertension.add_term('yes', lambda x: mf.triangular(x, 0.5, 1, 1.5))
        self.add_input_variable(hypertension)
        
        # Heart disease (0-1, treated as fuzzy)
        heart_disease = FuzzyVariable('heart_disease', (0, 1))
        heart_disease.add_term('no', lambda x: mf.triangular(x, -0.5, 0, 0.5))
        heart_disease.add_term('yes', lambda x: mf.triangular(x, 0.5, 1, 1.5))
        self.add_input_variable(heart_disease)
        
        # Output: Stroke Risk (0-100%)
        risk = FuzzyVariable('stroke_risk', (0, 100))
        risk.add_term('very_low', lambda x: mf.trapezoidal(x, 0, 0, 10, 25))
        risk.add_term('low', lambda x: mf.triangular(x, 15, 30, 45))
        risk.add_term('moderate', lambda x: mf.triangular(x, 35, 50, 65))
        risk.add_term('high', lambda x: mf.triangular(x, 55, 70, 85))
        risk.add_term('very_high', lambda x: mf.trapezoidal(x, 75, 90, 100, 100))
        self.add_output_variable(risk)
        
        # Output: Severity Level (0-10)
        severity = FuzzyVariable('severity', (0, 10))
        severity.add_term('mild', lambda x: mf.trapezoidal(x, 0, 0, 2, 4))
        severity.add_term('moderate', lambda x: mf.triangular(x, 3, 5, 7))
        severity.add_term('severe', lambda x: mf.trapezoidal(x, 6, 8, 10, 10))
        self.add_output_variable(severity)
    
    def _setup_rules(self):
        """Setup fuzzy rules untuk stroke prediction."""
        rule_id = 0
        
        # Age-based rules
        rule_id += 1
        self.add_rule(FuzzyRule(rule_id, [('age', 'young')], ('stroke_risk', 'very_low'), 0.9))
        
        rule_id += 1
        self.add_rule(FuzzyRule(rule_id, [('age', 'middle')], ('stroke_risk', 'low'), 0.7))
        
        rule_id += 1
        self.add_rule(FuzzyRule(rule_id, [('age', 'old')], ('stroke_risk', 'moderate'), 0.8))
        
        # Glucose-based rules
        rule_id += 1
        self.add_rule(FuzzyRule(rule_id, [('glucose', 'high')], ('stroke_risk', 'high'), 0.85))
        
        rule_id += 1
        self.add_rule(FuzzyRule(rule_id, [('glucose', 'normal')], ('stroke_risk', 'low'), 0.6))
        
        # Hypertension rules
        rule_id += 1
        self.add_rule(FuzzyRule(rule_id, [('hypertension', 'yes')], ('stroke_risk', 'high'), 0.9))
        
        # Heart disease rules
        rule_id += 1
        self.add_rule(FuzzyRule(rule_id, [('heart_disease', 'yes')], ('stroke_risk', 'high'), 0.9))
        
        # Combined rules - High risk combinations
        rule_id += 1
        self.add_rule(FuzzyRule(rule_id, 
            [('age', 'old'), ('hypertension', 'yes')], 
            ('stroke_risk', 'very_high'), 0.95))
        
        rule_id += 1
        self.add_rule(FuzzyRule(rule_id, 
            [('age', 'old'), ('heart_disease', 'yes')], 
            ('stroke_risk', 'very_high'), 0.95))
        
        rule_id += 1
        self.add_rule(FuzzyRule(rule_id, 
            [('age', 'old'), ('glucose', 'high')], 
            ('stroke_risk', 'very_high'), 0.9))
        
        rule_id += 1
        self.add_rule(FuzzyRule(rule_id, 
            [('hypertension', 'yes'), ('heart_disease', 'yes')], 
            ('stroke_risk', 'very_high'), 0.95))
        
        rule_id += 1
        self.add_rule(FuzzyRule(rule_id, 
            [('glucose', 'high'), ('bmi', 'obese')], 
            ('stroke_risk', 'high'), 0.85))
        
        # BMI rules
        rule_id += 1
        self.add_rule(FuzzyRule(rule_id, [('bmi', 'obese')], ('stroke_risk', 'moderate'), 0.7))
        
        rule_id += 1
        self.add_rule(FuzzyRule(rule_id, [('bmi', 'normal')], ('stroke_risk', 'very_low'), 0.5))
        
        # Low risk combinations
        rule_id += 1
        self.add_rule(FuzzyRule(rule_id, 
            [('age', 'young'), ('glucose', 'normal'), ('bmi', 'normal')], 
            ('stroke_risk', 'very_low'), 0.95))
        
        rule_id += 1
        self.add_rule(FuzzyRule(rule_id, 
            [('age', 'young'), ('hypertension', 'no'), ('heart_disease', 'no')], 
            ('stroke_risk', 'very_low'), 0.9))
        
        # Severity rules
        rule_id += 1
        self.add_rule(FuzzyRule(rule_id, 
            [('age', 'old'), ('hypertension', 'yes'), ('heart_disease', 'yes')], 
            ('severity', 'severe'), 0.95))
        
        rule_id += 1
        self.add_rule(FuzzyRule(rule_id, 
            [('age', 'old'), ('glucose', 'high')], 
            ('severity', 'severe'), 0.85))
        
        rule_id += 1
        self.add_rule(FuzzyRule(rule_id, 
            [('hypertension', 'yes'), ('glucose', 'high')], 
            ('severity', 'moderate'), 0.8))
        
        rule_id += 1
        self.add_rule(FuzzyRule(rule_id, 
            [('age', 'middle'), ('bmi', 'overweight')], 
            ('severity', 'moderate'), 0.6))
        
        rule_id += 1
        self.add_rule(FuzzyRule(rule_id, 
            [('age', 'young'), ('bmi', 'normal')], 
            ('severity', 'mild'), 0.8))
    
    def predict(self, age: float, glucose: float, bmi: float, 
                hypertension: int, heart_disease: int) -> Dict[str, any]:
        """
        Prediksi stroke risk untuk pasien.
        
        Parameters:
            age: Usia pasien
            glucose: Average glucose level
            bmi: Body Mass Index
            hypertension: 0 atau 1
            heart_disease: 0 atau 1
            
        Returns:
            Dictionary dengan stroke_risk, severity, dan interpretasi
        """
        inputs = {
            'age': age,
            'glucose': glucose,
            'bmi': bmi,
            'hypertension': float(hypertension),
            'heart_disease': float(heart_disease)
        }
        
        # Perform inference
        outputs = self.infer(inputs)
        
        # Get rule activations for explanation
        activations = self.get_rule_activations(inputs)
        active_rules = [(r, s) for r, s in activations if s > 0.1]
        active_rules.sort(key=lambda x: x[1], reverse=True)
        
        # Interpret results
        risk = outputs.get('stroke_risk', 0)
        severity = outputs.get('severity', 0)
        
        if risk < 20:
            risk_level = 'Very Low'
        elif risk < 40:
            risk_level = 'Low'
        elif risk < 60:
            risk_level = 'Moderate'
        elif risk < 80:
            risk_level = 'High'
        else:
            risk_level = 'Very High'
        
        if severity < 3.3:
            severity_level = 'Mild'
        elif severity < 6.6:
            severity_level = 'Moderate'
        else:
            severity_level = 'Severe'
        
        return {
            'stroke_risk_percentage': round(risk, 2),
            'risk_level': risk_level,
            'severity_score': round(severity, 2),
            'severity_level': severity_level,
            'top_rules': [(str(r), round(s, 3)) for r, s in active_rules[:5]],
            'input_memberships': self.fuzzify_inputs(inputs)
        }


# Utility functions for integration
def create_stroke_fuzzy_system() -> StrokeFuzzyExpertSystem:
    """Factory function untuk membuat stroke fuzzy system."""
    return StrokeFuzzyExpertSystem()


def fuzzy_predict_batch(system: StrokeFuzzyExpertSystem, 
                        data: List[Dict[str, float]]) -> List[Dict[str, any]]:
    """
    Prediksi untuk batch data.
    
    Parameters:
        system: Fuzzy expert system
        data: List of input dictionaries
        
    Returns:
        List of prediction results
    """
    results = []
    for patient in data:
        result = system.predict(
            age=patient['age'],
            glucose=patient['avg_glucose_level'],
            bmi=patient['bmi'],
            hypertension=patient['hypertension'],
            heart_disease=patient['heart_disease']
        )
        results.append(result)
    return results


if __name__ == "__main__":
    # Test the fuzzy system
    system = create_stroke_fuzzy_system()
    
    # Test case 1: High risk patient
    result = system.predict(
        age=75,
        glucose=200,
        bmi=32,
        hypertension=1,
        heart_disease=1
    )
    print("High Risk Patient:")
    print(f"  Risk: {result['stroke_risk_percentage']}% ({result['risk_level']})")
    print(f"  Severity: {result['severity_score']}/10 ({result['severity_level']})")
    print()
    
    # Test case 2: Low risk patient
    result = system.predict(
        age=30,
        glucose=90,
        bmi=22,
        hypertension=0,
        heart_disease=0
    )
    print("Low Risk Patient:")
    print(f"  Risk: {result['stroke_risk_percentage']}% ({result['risk_level']})")
    print(f"  Severity: {result['severity_score']}/10 ({result['severity_level']})")
