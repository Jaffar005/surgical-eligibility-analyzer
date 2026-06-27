import streamlit as st
import google.generativeai as genai
import PyPDF2
import json
import re
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import os

# ---------------- CONFIGURE GEMINI ----------------
# Use Streamlit secrets first, fallback to environment variable
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ---------------- INITIALIZE SESSION STATE ----------------
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:vemb
    st.session_state.scaler = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model_accuracy' not in st.session_state:
    st.session_state.model_accuracy = None

# ---------------- TRAIN RANDOM FOREST FROM CSV ----------------
def train_random_forest_from_csv(csv_file=None):
    try:
        if csv_file:
            # Load from uploaded CSV
            df = pd.read_csv(csv_file)
        else:
            # Load from sample.txt file
            if os.path.exists('sample.txt'):
                df = pd.read_csv('sample.txt')
            else:
                st.error("sample.txt file not found. Please upload a CSV file.")
                return None, None
        
        # Check if required columns exist
        required_features = ['age', 'systolic_bp', 'diastolic_bp', 'has_diabetes', 
                            'has_hypertension', 'has_heart_disease', 'has_cancer', 
                            'has_kidney_disease', 'symptom_count', 'bmi', 'previous_surgery']
        
        if not all(col in df.columns for col in required_features):
            st.error(f"CSV must contain columns: {required_features}")
            return None, None
        
        if 'eligibility' not in df.columns:
            st.error("CSV must contain 'eligibility' column")
            return None, None
        
        # Prepare features and target
        X = df[required_features]
        y = df['eligibility']
        
        # Encode target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Train Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        with st.spinner("Training model..."):
            rf_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(rf_model, X_scaled, y_encoded, cv=5)
        
        # Store model in session state
        st.session_state.model = rf_model
        st.session_state.scaler = scaler
        st.session_state.label_encoder = le
        st.session_state.feature_columns = required_features
        st.session_state.model_trained = True
        st.session_state.model_accuracy = {
            'test_accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores,
            'feature_importance': dict(zip(required_features, rf_model.feature_importances_)),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, target_names=le.classes_),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'total_samples': len(df)
        }
        
        # Save model to disk
        joblib.dump(rf_model, 'random_forest_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(le, 'label_encoder.pkl')
        joblib.dump(required_features, 'feature_columns.pkl')
        
        return accuracy, cv_scores.mean()
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None

# ---------------- LOAD OR TRAIN MODEL ----------------
def load_or_train_model():
    try:
        # Try to load pre-trained model
        if os.path.exists('random_forest_model.pkl'):
            model = joblib.load('random_forest_model.pkl')
            scaler = joblib.load('scaler.pkl')
            le = joblib.load('label_encoder.pkl')
            feature_columns = joblib.load('feature_columns.pkl')
            
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.label_encoder = le
            st.session_state.feature_columns = feature_columns
            st.session_state.model_trained = True
            
            return True
        return False
    except:
        return False

# ---------------- PDF TEXT EXTRACTION ----------------
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        if hasattr(pdf_file, 'read'):
            pdf_reader = PyPDF2.PdfReader(pdf_file)
        else:
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file))
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        return ""

# ---------------- MERGE MULTIPLE PDFS ----------------
def merge_pdfs_text(pdf_files):
    all_text = []
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        if text:
            all_text.append(f"--- Document: {pdf_file.name} ---\n{text}\n")
    if not all_text:
        return ""
    return "\n".join(all_text)

# ---------------- EXTRACT FEATURES FROM TEXT ----------------
def extract_features_from_text(text):
    # Age extraction
    age_match = re.findall(r'Age[:\s]*(\d+)', text, re.IGNORECASE)
    age = int(age_match[0]) if age_match else 50
    
    # BP extraction
    bp_match = re.findall(r'(\d{2,3})/(\d{2,3})', text)
    systolic_bp = int(bp_match[0][0]) if bp_match else 120
    diastolic_bp = int(bp_match[0][1]) if bp_match else 80
    
    # Disease detection
    text_lower = text.lower()
    has_diabetes = 1 if 'diabetes' in text_lower else 0
    has_hypertension = 1 if 'hypertension' in text_lower or 'high bp' in text_lower or 'high blood pressure' in text_lower else 0
    has_heart_disease = 1 if 'heart' in text_lower or 'cardiac' in text_lower or 'coronary' in text_lower else 0
    has_cancer = 1 if 'cancer' in text_lower or 'malignant' in text_lower or 'tumor' in text_lower else 0
    has_kidney_disease = 1 if 'kidney' in text_lower or 'renal' in text_lower else 0
    
    # Symptom count
    symptom_keywords = ['pain', 'fever', 'cough', 'fatigue', 'nausea', 'dizziness', 
                        'shortness of breath', 'chest pain', 'headache', 'vomiting', 
                        'diarrhea', 'swelling', 'weakness']
    symptom_count = sum(1 for keyword in symptom_keywords if keyword in text_lower)
    
    # BMI extraction
    bmi_match = re.findall(r'BMI[:\s]*(\d+\.?\d*)', text, re.IGNORECASE)
    bmi = float(bmi_match[0]) if bmi_match else 25.0
    
    # Previous surgery
    previous_surgery = 1 if 'previous surgery' in text_lower or 'prior surgery' in text_lower or 'surgical history' in text_lower else 0
    
    return {
        'age': age,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'has_diabetes': has_diabetes,
        'has_hypertension': has_hypertension,
        'has_heart_disease': has_heart_disease,
        'has_cancer': has_cancer,
        'has_kidney_disease': has_kidney_disease,
        'symptom_count': symptom_count,
        'bmi': bmi,
        'previous_surgery': previous_surgery
    }

# ---------------- PREDICT WITH RANDOM FOREST ----------------
def predict_with_model(features):
    if not st.session_state.model_trained:
        return None
    
    # Create dataframe with correct feature order
    df = pd.DataFrame([features])
    df = df[st.session_state.feature_columns]
    
    # Scale features
    df_scaled = st.session_state.scaler.transform(df)
    
    # Predict
    prediction_encoded = st.session_state.model.predict(df_scaled)[0]
    prediction_proba = st.session_state.model.predict_proba(df_scaled)[0]
    
    # Decode prediction
    eligibility = st.session_state.label_encoder.inverse_transform([prediction_encoded])[0]
    
    # Get confidence
    confidence = prediction_proba[prediction_encoded] * 100
    
    # Get risk level based on prediction
    if eligibility == "Eligible":
        risk_level = "Low"
    elif eligibility == "Not Eligible":
        risk_level = "High"
    else:
        risk_level = "Medium"
    
    return {
        'eligibility': eligibility,
        'risk_level': risk_level,
        'confidence': confidence,
        'probabilities': {
            label: float(prob) 
            for label, prob in zip(st.session_state.label_encoder.classes_, prediction_proba)
        }
    }

# ---------------- AI ANALYSIS ----------------
def analyze_patient_with_ai(text, features):
    prompt = f"""
You are a medical AI assistant. Provide clinical reasoning based on the extracted features.

Patient Clinical Data:
- Age: {features['age']}
- Blood Pressure: {features['systolic_bp']}/{features['diastolic_bp']}
- Diabetes: {'Yes' if features['has_diabetes'] else 'No'}
- Hypertension: {'Yes' if features['has_hypertension'] else 'No'}
- Heart Disease: {'Yes' if features['has_heart_disease'] else 'No'}
- Cancer: {'Yes' if features['has_cancer'] else 'No'}
- Kidney Disease: {'Yes' if features['has_kidney_disease'] else 'No'}
- Symptom Count: {features['symptom_count']}
- BMI: {features['bmi']:.1f}
- Previous Surgery: {'Yes' if features['previous_surgery'] else 'No'}

Provide a brief clinical reasoning for surgical eligibility assessment.
Keep it concise (2-3 sentences).
"""
    
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except:
        return "Based on clinical parameters and risk factors assessment."

# ---------------- DISPLAY CONFUSION MATRIX ----------------
def plot_confusion_matrix(cm, classes):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Add text annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = ax.text(j, i, cm[i, j],
                          ha="center", va="center", 
                          color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    return fig

# ---------------- STREAMLIT UI ----------------
def main():
    st.set_page_config(
        page_title="Surgical Eligibility Analyzer", 
        page_icon="⚕️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inject Custom CSS for Premium UI
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        /* Main container and text */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        h1, h2, h3 {
            color: #0f4c75;
            font-weight: 600;
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #f8f9fa;
        }
        .sidebar-header {
            font-size: 1.2rem;
            font-weight: 600;
            color: #1E3A8A;
            margin-bottom: 1rem;
        }
        
        /* Button styling */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            border: 1px solid #e2e8f0;
            transition: all 0.3s ease;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            color: white;
            border: none;
            box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.3);
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px -2px rgba(59, 130, 246, 0.4);
        }
        
        /* Metric Cards */
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 15px 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            transition: all 0.3s ease;
        }
        div[data-testid="stMetric"]:hover {
            box-shadow: 0 8px 16px rgba(0,0,0,0.08);
            transform: translateY(-3px);
            border-color: #3b82f6;
        }
        div[data-testid="stMetricValue"] {
            color: #1e3a8a;
            font-weight: 700;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background-color: #f8fafc;
            border-radius: 8px;
            font-weight: 500;
            color: #1e293b;
        }
        
        /* File Uploader Dropzone */
        section[data-testid="stFileUploadDropzone"] {
            border: 2px dashed #cbd5e1;
            border-radius: 12px;
            background-color: #f8fafc;
            padding: 2rem;
            transition: all 0.3s ease;
        }
        section[data-testid="stFileUploadDropzone"]:hover {
            border-color: #3b82f6;
            background-color: #eff6ff;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center; color: #1E3A8A; font-weight: 800; font-size: 3rem; margin-bottom: 0;'>⚕️AI-Based Surgical Eligibility Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748B; font-size: 1.2rem; font-weight: 500; margin-bottom: 2rem;'>Intelligent Clinical Support System for Operative Risk Assessment</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for model training
    with st.sidebar:
        st.markdown("<div class='sidebar-header'>⚙️ Control Panel</div>", unsafe_allow_html=True)
        st.header("📊 Model Training")
        
        # Option to upload CSV or use sample.txt
        train_option = st.radio(
            "Training Data Source",
            ["Use sample.txt", "Upload CSV File"]
        )
        
        if train_option == "Upload CSV File":
            uploaded_csv = st.file_uploader("Upload Training CSV", type=["csv"])
            if uploaded_csv:
                if st.button("🚀 Train with Uploaded CSV", type="primary"):
                    accuracy, cv_mean = train_random_forest_from_csv(uploaded_csv)
                    if accuracy:
                        st.success(f"✅ Model trained! Test Accuracy: {accuracy*100:.2f}% | CV Score: {cv_mean*100:.2f}%")
                        st.rerun()
        else:
            if os.path.exists('sample.txt'):
                if st.button("🚀 Train with sample.txt", type="primary"):
                    accuracy, cv_mean = train_random_forest_from_csv()
                    if accuracy:
                        st.success(f"✅ Model trained! Test Accuracy: {accuracy*100:.2f}% | CV Score: {cv_mean*100:.2f}%")
                        st.rerun()
            else:
                st.error("sample.txt file not found. Please upload a CSV file.")
        
        st.markdown("---")
        
        # Display model performance if trained
        if st.session_state.model_trained and st.session_state.model_accuracy:
            st.header("📈 Model Performance")
            st.metric("Test Accuracy", f"{st.session_state.model_accuracy['test_accuracy']*100:.2f}%")
            st.metric("Cross-Validation Score", f"{st.session_state.model_accuracy['cv_mean']*100:.2f}%")
            st.metric("Training Samples", st.session_state.model_accuracy['train_samples'])
            st.metric("Test Samples", st.session_state.model_accuracy['test_samples'])
            
            with st.expander("📊 Feature Importance"):
                importance = st.session_state.model_accuracy['feature_importance']
                sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                for feature, imp in sorted_importance:
                    st.progress(imp, text=f"{feature}: {imp*100:.1f}%")
            
            with st.expander("📋 Classification Report"):
                st.text(st.session_state.model_accuracy['classification_report'])
            
            with st.expander("🎯 Confusion Matrix"):
                cm = st.session_state.model_accuracy['confusion_matrix']
                classes = st.session_state.label_encoder.classes_
                fig = plot_confusion_matrix(cm, classes)
                st.pyplot(fig)
            
            with st.expander("📊 Cross-Validation Scores"):
                for i, score in enumerate(st.session_state.model_accuracy['cv_scores'], 1):
                    st.write(f"Fold {i}: {score*100:.2f}%")
                st.write(f"Mean: {st.session_state.model_accuracy['cv_mean']*100:.2f}%")
                st.write(f"Std Dev: {st.session_state.model_accuracy['cv_std']*100:.2f}%")
    
    # Main content
    if not st.session_state.model_trained:
        st.warning("⚠️ Model not trained. Please train the model using the sidebar.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Upload Medical Reports (PDF)", 
            type=["pdf"], 
            accept_multiple_files=True,
            help="Upload one or multiple PDF files"
        )
        
        manual_input = st.text_area("Or paste medical report manually", height=150)
    
    
    if uploaded_files or manual_input:
        st.markdown("---")
        
        if uploaded_files:
            with st.spinner(f"Processing {len(uploaded_files)} PDF(s)..."):
                text = merge_pdfs_text(uploaded_files)
                if text:
                    st.success(f"✅ Successfully processed {len(uploaded_files)} file(s)")
                else:
                    st.error("❌ Failed to extract text from PDFs")
        else:
            text = manual_input
            if text:
                st.info("📝 Manual input detected")
        
        if text:
            with st.expander("📄 View Extracted Text"):
                st.text(text[:1000] + ("..." if len(text) > 1000 else ""))
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔍 Run Full Clinical Analysis", type="primary", use_container_width=True):
                with st.spinner("Extracting clinical features from text..."):
                    features = extract_features_from_text(text)
                
                with st.spinner("Predicting with model..."):
                    prediction = predict_with_model(features)
                
                with st.spinner("Generating clinical reasoning..."):
                    reasoning = analyze_patient_with_ai(text, features)
                
                if prediction:
                    st.markdown("---")
                    st.markdown("<h2 style='color: #1E3A8A; margin-bottom: 20px;'>📊 Comprehensive Analysis Results</h2>", unsafe_allow_html=True)
                    
                    # Display features
                    with st.expander("📋 Extracted Clinical Features"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Age", features['age'])
                            st.metric("BMI", f"{features['bmi']:.1f}")
                            st.metric("BP", f"{features['systolic_bp']}/{features['diastolic_bp']}")
                        with col2:
                            st.metric("Diabetes", "Yes" if features['has_diabetes'] else "No")
                            st.metric("Hypertension", "Yes" if features['has_hypertension'] else "No")
                            st.metric("Heart Disease", "Yes" if features['has_heart_disease'] else "No")
                        with col3:
                            st.metric("Cancer", "Yes" if features['has_cancer'] else "No")
                            st.metric("Kidney Disease", "Yes" if features['has_kidney_disease'] else "No")
                            st.metric("Previous Surgery", "Yes" if features['previous_surgery'] else "No")
                    
                    # Display prediction results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        risk_color = "🟢" if prediction['risk_level'] == "Low" else "🟡" if prediction['risk_level'] == "Medium" else "🔴"
                        st.metric("Risk Level", f"{risk_color} {prediction['risk_level']}")
                        
                    
                    with col2:
                        if prediction['eligibility'] == "Eligible":
                            st.success(f"**Surgical Eligibility:** {prediction['eligibility']} ✅")
                        elif prediction['eligibility'] == "Not Eligible":
                            st.error(f"**Surgical Eligibility:** {prediction['eligibility']} ❌")
                        else:
                            st.warning(f"**Surgical Eligibility:** {prediction['eligibility']} ⚠️")
                    
                    # Show probability distribution
                    st.subheader("📊 Prediction Probability Distribution")
                    prob_cols = st.columns(len(prediction['probabilities']))
                    for idx, (label, prob) in enumerate(prediction['probabilities'].items()):
                        with prob_cols[idx]:
                            st.progress(prob, text=f"{label}\n{prob*100:.1f}%")
                    
                    st.markdown("---")
                    st.subheader("📌 Clinical Reasoning")
                    st.write(reasoning)
                    
                    st.warning("⚠️ **Disclaimer:** AI-assisted decision based on Random Forest model. Final surgical decision requires consultation with a qualified healthcare professional.")
                else:
                    st.error("Prediction failed. Please try again.")
    
    # Footer
    st.markdown("---")
    if st.session_state.model_accuracy:
        st.caption(f"Surgical Eligibility Analyzer v3.0 | Random Forest Model | Accuracy: {st.session_state.model_accuracy['test_accuracy']*100:.1f}% (validated on test data)")
    else:
        st.caption("Surgical Eligibility Analyzer v3.0 | Random Forest Model")

if __name__ == "__main__":
    # Automatically load model if available
    if not st.session_state.model_trained:
        load_or_train_model()
    main()