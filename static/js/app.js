/**
 * NeuroFuzzy Medical Diagnosis System
 * Frontend JavaScript Application
 * 
 * Handles:
 * - Form submission and validation
 * - API communication
 * - Results visualization
 * - Charts rendering
 * - Dynamic updates
 */

// ============================================
// Configuration
// ============================================
const API_BASE = '';

// Chart color palette - Medical Green Theme
const COLORS = {
    primary: '#059669',
    primaryLight: '#10b981',
    secondary: '#0d9488',
    accent: '#14b8a6',
    success: '#22c55e',
    warning: '#f59e0b',
    danger: '#ef4444',
    info: '#0ea5e9',
    dark: '#064e3b',
    text: '#064e3b',
    muted: '#6b7280',
    background: '#f0fdf4'
};

// Chart.js default configuration for light theme
Chart.defaults.color = COLORS.muted;
Chart.defaults.font.family = "'Inter', sans-serif";

// ============================================
// DOM Elements
// ============================================
const diagnosisForm = document.getElementById('diagnosis-form');
const analyzeBtn = document.getElementById('analyze-btn');
const resultsContent = document.getElementById('results-content');

// ============================================
// Initialization
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();
    setupNavigation();
});

async function initializeApp() {
    try {
        // Check system health
        const health = await fetchAPI('/api/health');
        updateSystemStatus(health);
        
        // Load initial data
        await Promise.all([
            loadMetrics(),
            loadEDA()
        ]);
    } catch (error) {
        console.error('Initialization error:', error);
        updateSystemStatus({ status: 'error' });
    }
}

function setupEventListeners() {
    // Form submission
    diagnosisForm.addEventListener('submit', handleDiagnosis);
    
    // Input validation and real-time feedback
    const inputs = diagnosisForm.querySelectorAll('input[type="number"]');
    inputs.forEach(input => {
        input.addEventListener('input', validateInput);
    });
    
    // Glucose level indicator
    const glucoseInput = document.getElementById('glucose');
    glucoseInput.addEventListener('input', updateGlucoseIndicator);
}

function setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            navLinks.forEach(l => l.classList.remove('active'));
            e.target.classList.add('active');
        });
    });
    
    // Smooth scrolling
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
}

// ============================================
// API Functions
// ============================================
async function fetchAPI(endpoint, options = {}) {
    const response = await fetch(`${API_BASE}${endpoint}`, {
        headers: { 'Content-Type': 'application/json' },
        ...options
    });
    
    if (!response.ok) {
        throw new Error(`API Error: ${response.statusText}`);
    }
    
    return response.json();
}

// ============================================
// Diagnosis Handler
// ============================================
async function handleDiagnosis(e) {
    e.preventDefault();
    
    // Set loading state
    analyzeBtn.classList.add('loading');
    analyzeBtn.innerHTML = '<span>⏳ Menganalisa...</span>';
    
    try {
        const formData = {
            age: parseFloat(document.getElementById('age').value),
            avg_glucose_level: parseFloat(document.getElementById('glucose').value),
            bmi: parseFloat(document.getElementById('bmi').value),
            hypertension: document.getElementById('hypertension').checked ? 1 : 0,
            heart_disease: document.getElementById('heart_disease').checked ? 1 : 0
        };
        
        const result = await fetchAPI('/predict', {
            method: 'POST',
            body: JSON.stringify(formData)
        });
        
        displayResults(result);
        
    } catch (error) {
        console.error('Diagnosis error:', error);
        displayError(error.message);
    } finally {
        analyzeBtn.classList.remove('loading');
        analyzeBtn.innerHTML = '<span>🔍 Analisis Resiko</span>';
    }
}

function displayResults(result) {
    const riskClass = result.risk_level.toLowerCase().replace(' ', '-');
    const gaugeColor = getColorForRisk(result.risk_level);
    
    resultsContent.innerHTML = `
        <div class="result-display">
            <div class="risk-gauge">
                <div class="gauge-value" style="color: ${gaugeColor}">
                    ${result.stroke_probability}%
                </div>
                <div class="gauge-label">Resiko Stroke</div>
                <span class="risk-level ${riskClass}">${result.risk_level}</span>
            </div>
            
            <div class="result-details">
                <div class="detail-item">
                    <label>Skor Resiko (Fuzzy)</label>
                    <span>${result.fuzzy_risk}%</span>
                </div>
                <div class="detail-item">
                    <label>Neural Network</label>
                    <span>${result.nn_probability !== null ? result.nn_probability + '%' : 'N/A'}</span>
                </div>
                <div class="detail-item">
                    <label>Skor Severity</label>
                    <span>${result.severity_score}/10</span>
                </div>
                <div class="detail-item">
                    <label>Level Severity</label>
                    <span>${result.severity_level}</span>
                </div>
                <div class="detail-item">
                    <label>Prediksi</label>
                    <span>${result.prediction === 1 ? 'Resiko' : '✅ Low Risk'}</span>
                </div>
                <div class="detail-item">
                    <label>Confidence</label>
                    <span>${result.confidence}%</span>
                </div>
            </div>
        </div>
    `;
}

function displayError(message) {
    resultsContent.innerHTML = `
        <div class="error-display" style="text-align: center; padding: 2rem; color: ${COLORS.danger};">
            <div style="font-size: 3rem; margin-bottom: 1rem;">⚠️</div>
            <p>Error: ${message}</p>
            <p style="color: ${COLORS.muted}; font-size: 0.875rem; margin-top: 0.5rem;">TOlong coba lagi.</p>
        </div>
    `;
}

function getColorForRisk(riskLevel) {
    const colorMap = {
        'Sangat Rendah': COLORS.success,
        'Rendah': COLORS.accent,
        'Cukup Tinggi': COLORS.warning,
        'TInggi': COLORS.danger,
        'Sangat Tinggi': '#fca5a5'
    };
    return colorMap[riskLevel] ||    COLORS.primary;
}

// ============================================
// Metrics Loading
// ============================================
async function loadMetrics() {
    try {
        const metrics = await fetchAPI('/metrics');
        
        // Update hero stats
        document.getElementById('accuracy-stat').textContent = (metrics.accuracy * 100).toFixed(1) + '%';
        document.getElementById('auc-stat').textContent = metrics.auc_roc.toFixed(3);
        
        // Update metrics cards
        updateMetricCard('accuracy', metrics.accuracy);
        updateMetricCard('sensitivity', metrics.sensitivity);
        updateMetricCard('specificity', metrics.specificity);
        updateMetricCard('auc', metrics.auc_roc);
        
        // Update confusion matrix
        if (metrics.confusion_matrix) {
            document.getElementById('cm-tn').textContent = metrics.confusion_matrix.TN;
            document.getElementById('cm-fp').textContent = metrics.confusion_matrix.FP;
            document.getElementById('cm-fn').textContent = metrics.confusion_matrix.FN;
            document.getElementById('cm-tp').textContent = metrics.confusion_matrix.TP;
        }
        
        // Draw ROC curve
        if (metrics.roc_curve) {
            drawROCCurve(metrics.roc_curve);
        }
        
    } catch (error) {
        console.error('Gagal Memuat Metric:', error);
    }
}

function updateMetricCard(metric, value) {
    const valueEl = document.getElementById(`metric-${metric}`);
    const barEl = document.getElementById(`${metric}-bar`);
    
    if (valueEl) {
        valueEl.textContent = (value * 100).toFixed(1) + '%';
    }
    if (barEl) {
        barEl.style.width = (value * 100) + '%';
    }
}

// ============================================
// EDA Loading
// ============================================
async function loadEDA() {
    try {
        const eda = await fetchAPI('/eda');
        
        // Update stats
        document.getElementById('records-stat').textContent = eda.total_records || '--';
        document.getElementById('eda-stroke-cases').textContent = eda.stroke_cases || '--';
        document.getElementById('eda-no-stroke').textContent = eda.no_stroke_cases || '--';
        document.getElementById('eda-stroke-rate').textContent = (eda.stroke_rate || 0) + '%';
        document.getElementById('eda-total').textContent = eda.total_records || '--';
        
        // Draw charts
        if (eda.age_stats) drawAgeChart(eda.age_stats);
        if (eda.stroke_by_age) drawStrokeAgeChart(eda.stroke_by_age);
        if (eda.risk_factors) drawRiskFactorsChart(eda.risk_factors);
        if (eda.glucose_stats) drawGlucoseChart(eda.glucose_stats);
        
    } catch (error) {
        console.error('Gagal memuat EDA:', error);
    }
}

// ============================================
// Chart Drawing Functions
// ============================================
function drawAgeChart(ageStats) {
    const ctx = document.getElementById('ageChart')?.getContext('2d');
    if (!ctx || !ageStats.bins) return;
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ageStats.bins.map(b => b.range),
            datasets: [{
                label: 'Count',
                data: ageStats.bins.map(b => b.count),
                backgroundColor: COLORS.primary + '80',
                borderColor: COLORS.primary,
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: { grid: { display: false } },
                y: { beginAtZero: true, grid: { color: 'rgba(5, 150, 105, 0.1)' } }
            }
        }
    });
}

function drawStrokeAgeChart(strokeByAge) {
    const ctx = document.getElementById('strokeAgeChart')?.getContext('2d');
    if (!ctx) return;
    
    const labels = Object.keys(strokeByAge).map(k => k.charAt(0).toUpperCase() + k.slice(1));
    const strokeData = Object.values(strokeByAge).map(v => v.stroke);
    const noStrokeData = Object.values(strokeByAge).map(v => v.no_stroke);
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Stroke',
                    data: strokeData,
                    backgroundColor: COLORS.danger + '80',
                    borderColor: COLORS.danger,
                    borderWidth: 1,
                    borderRadius: 4
                },
                {
                    label: 'No Stroke',
                    data: noStrokeData,
                    backgroundColor: COLORS.success + '80',
                    borderColor: COLORS.success,
                    borderWidth: 1,
                    borderRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom' }
            },
            scales: {
                x: { grid: { display: false } },
                y: { beginAtZero: true, grid: { color: 'rgba(5, 150, 105, 0.1)' } }
            }
        }
    });
}

function drawRiskFactorsChart(riskFactors) {
    const ctx = document.getElementById('riskFactorsChart')?.getContext('2d');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Hypertension', 'Heart Disease', 'Normal'],
            datasets: [{
                data: [
                    riskFactors.hypertension?.with_factor || 0,
                    riskFactors.heart_disease?.with_factor || 0,
                    Math.max(0, (riskFactors.hypertension?.without_factor || 0) - (riskFactors.heart_disease?.with_factor || 0))
                ],
                backgroundColor: [COLORS.danger, COLORS.warning, COLORS.success],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom' }
            }
        }
    });
}

function drawGlucoseChart(glucoseStats) {
    const ctx = document.getElementById('glucoseChart')?.getContext('2d');
    if (!ctx || !glucoseStats.bins) return;
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: glucoseStats.bins.map(b => b.range),
            datasets: [{
                label: 'Glucose Distribution',
                data: glucoseStats.bins.map(b => b.count),
                fill: true,
                backgroundColor: COLORS.accent + '30',
                borderColor: COLORS.accent,
                borderWidth: 2,
                tension: 0.4,
                pointRadius: 4,
                pointBackgroundColor: COLORS.accent
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: { grid: { display: false } },
                y: { beginAtZero: true, grid: { color: 'rgba(5, 150, 105, 0.1)' } }
            }
        }
    });
}

function drawROCCurve(rocData) {
    const ctx = document.getElementById('rocChart')?.getContext('2d');
    if (!ctx || !rocData.fpr || !rocData.tpr) return;
    
    // Prepare data points
    const data = rocData.fpr.map((fpr, i) => ({ x: fpr, y: rocData.tpr[i] }));
    
    new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'ROC Curve',
                    data: data,
                    borderColor: COLORS.primary,
                    backgroundColor: COLORS.primary + '20',
                    fill: true,
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1
                },
                {
                    label: 'Random (AUC = 0.5)',
                    data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
                    borderColor: COLORS.muted,
                    borderDash: [5, 5],
                    borderWidth: 1,
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom' }
            },
            scales: {
                x: {
                    type: 'linear',
                    min: 0,
                    max: 1,
                    title: { display: true, text: 'False Positive Rate' },
                    grid: { color: 'rgba(5, 150, 105, 0.1)' }
                },
                y: {
                    min: 0,
                    max: 1,
                    title: { display: true, text: 'True Positive Rate' },
                    grid: { color: 'rgba(5, 150, 105, 0.1)' }
                }
            }
        }
    });
}

// ============================================
// Utility Functions
// ============================================
function updateSystemStatus(health) {
    const statusEl = document.getElementById('system-status');
    const statusDot = document.querySelector('.status-dot');
    
    if (health.status === 'healthy') {
        statusEl.textContent = 'System Ready';
        statusDot.style.background = COLORS.success;
    } else {
        statusEl.textContent = 'System Error';
        statusDot.style.background = COLORS.danger;
    }
}

function validateInput(e) {
    const input = e.target;
    const value = parseFloat(input.value);
    const min = parseFloat(input.min);
    const max = parseFloat(input.max);
    
    if (value < min || value > max) {
        input.style.borderColor = COLORS.danger;
    } else {
        input.style.borderColor = '';
    }
}

function updateGlucoseIndicator(e) {
    const value = parseFloat(e.target.value);
    const bar = document.querySelector('.glucose-bar');
    
    if (bar) {
        const spans = bar.querySelectorAll('span');
        spans.forEach(span => span.style.fontWeight = 'normal');
        
        if (value < 100) {
            spans[0].style.fontWeight = 'bold';
        } else if (value < 140) {
            spans[1].style.fontWeight = 'bold';
        } else {
            spans[2].style.fontWeight = 'bold';
        }
    }
}

// ============================================
// Export for testing
// ============================================
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        fetchAPI,
        handleDiagnosis,
        displayResults,
        getColorForRisk
    };
}
