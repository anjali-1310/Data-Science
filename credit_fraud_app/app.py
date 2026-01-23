import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("💳 Credit Card Fraud Detection App")
st.write("Upload a CSV file with transaction data to detect fraudulent transactions.")

# ---------------------------
# Load trained model (SAFE)
# ---------------------------
st.write("⏳ Loading model...")

try:
    with open("credit_risk_model.pkl", "rb") as f:
        model = pickle.load(f)
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error("❌ Model loading failed")
    st.error(str(e))
    st.stop()

# ---------------------------
# Expected feature columns
# ---------------------------
EXPECTED_COLUMNS = [
    'Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
    'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
    'V21','V22','V23','V24','V25','V26','V27','V28','Amount'
]

uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.subheader("📊 Uploaded Data Preview")
        st.dataframe(data.head())

        # Check columns
        missing_cols = [col for col in EXPECTED_COLUMNS if col not in data.columns]
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
            st.stop()

        X = data[EXPECTED_COLUMNS]

        if st.button("🔍 Predict Fraud"):
            try:
                preds = model.predict(X)

                # Probabilities (if supported)
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X)[:, 1]
                else:
                    probs = np.zeros(len(preds))

                result_df = data.copy()
                result_df['Fraud_Prediction'] = preds
                result_df['Fraud_Probability'] = probs
                result_df['Fraud_Prediction_Label'] = result_df['Fraud_Prediction'].map(
                    {0: 'Genuine', 1: 'Fraud'}
                )

                fraud_count = (preds == 1).sum()
                genuine_count = (preds == 0).sum()

                col1, col2 = st.columns(2)
                col1.metric("🟢 Genuine Transactions", genuine_count)
                col2.metric("🔴 Fraudulent Transactions", fraud_count)

                st.subheader("✅ Prediction Results")
                st.dataframe(result_df.head(50))

                # Download results
                csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Predictions CSV",
                    data=csv,
                    file_name="fraud_predictions.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error("❌ Prediction failed")
                st.error(str(e))

    except Exception as e:
        st.error("⚠️ Error processing file")
        st.error(str(e))

st.markdown("---")
st.caption("Built with Streamlit | Credit Card Fraud Detection Model")

