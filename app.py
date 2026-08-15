import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

# Page configuration
st.set_page_config(page_title="Credit Default Predictor", layout="wide")
st.title("Credit Card Default Classification App")
st.markdown("Upload your **test_data.csv** and select a model to view its performance metrics.")

# Dictionary to map dropdown names to the saved file names
model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl"
}

# Feature a: Dataset upload option
uploaded_file = st.file_uploader("Upload Test Data (CSV format only)", type="csv")

# Feature b: Model selection dropdown[cite: 1]
selected_model = st.selectbox("Select ML Model", list(model_files.keys()))

if uploaded_file is not None:
    # Read data
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())
    
    if 'Target' not in df.columns:
        st.error("Error: The uploaded CSV must contain a 'Target' column.")
    else:
        X_test = df.drop(columns=['Target'])
        y_test = df['Target']
        
        try:
            # Load the scaler
            scaler = joblib.load('model/scaler.pkl')
            X_test_scaled = scaler.transform(X_test)
            
            # Load the chosen model
            model_path = os.path.join('model', model_files[selected_model])
            model = joblib.load(model_path)
            
            # Generate Predictions
            y_pred = model.predict(X_test_scaled)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_proba = [0] * len(y_test)
            
            st.divider()
            
            # Feature c: Display of evaluation metrics[cite: 1]
            st.subheader(f"Evaluation Metrics: {selected_model}")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
            col1.metric("AUC Score", f"{roc_auc_score(y_test, y_proba):.4f}")
            col2.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.4f}")
            col2.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
            col3.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")
            col3.metric("MCC Score", f"{matthews_corrcoef(y_test, y_pred):.4f}")
            
            st.divider()
            
            # Feature d: Confusion matrix and classification report[cite: 1]
            st.subheader("Model Diagnostics")
            
            diag_col1, diag_col2 = st.columns(2)
            
            with diag_col1:
                st.markdown("**Confusion Matrix**")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')
                st.pyplot(fig)
                
            with diag_col2:
                st.markdown("**Classification Report**")
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                st.dataframe(pd.DataFrame(report).transpose())
                
        except Exception as e:
            st.error(f"Failed to load the model or process the data. Error: {e}")
else:
    st.info("Please upload the test_data.csv file to proceed.")