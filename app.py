import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

# Page config
st.set_page_config(page_title="Diabetes Prediction", layout="wide")
st.title("🏥 Diabetes Prediction Model")
st.markdown("Enter patient details to predict diabetes risk")

# Load Pima Indian Diabetes dataset (small, so we can train on-the-fly)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    data = pd.read_csv(url, names=column_names)
    return data

# Train model function
@st.cache_resource
def train_model():
    data = load_data()
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=500)
    model.fit(X_scaled, y)
    return model, scaler

# Load trained model and scaler
model, scaler = train_model()

# Create input columns
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 4, key="preg")
    glucose = st.slider("Glucose (mg/dL)", 0, 200, 120, key="gluc")
    blood_pressure = st.slider("Blood Pressure (mmHg)", 0, 122, 72, key="bp")

with col2:
    skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 29, key="skin")
    insulin = st.slider("Insulin (mu U/ml)", 0, 846, 125, key="ins")
    bmi = st.slider("BMI", 0.0, 67.1, 32.0, 1.0, key="bmi")

with col3:
    diabetes_pedigree = st.slider("Diabetes Pedigree Function", 0.078, 2.420, 0.47, key="dpf")
    age = st.slider("Age (years)", 21, 81, 30, key="age")

# Prediction
if st.button("🔮 Predict Diabetes Risk", type="primary", use_container_width=True):
    # Prepare input data
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                           insulin, bmi, diabetes_pedigree, age]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    # Display result
    col1, col2, col3 = st.columns([1, 2, 1])
    
    if prediction == 1:
        col2.error("🚨 **DIABETIC** (Risk: {:.1%})".format(probability))
        st.markdown("**⚠️ Recommendations:** Consult a doctor immediately, monitor blood sugar, consider lifestyle changes.")
    else:
        col2.success("✅ **NON-DIABETIC** (Risk: {:.1%})".format(probability))
        st.markdown("**💡 Tips:** Maintain healthy lifestyle, regular checkups recommended.")

# Model metrics sidebar
with st.sidebar:
    st.header("📊 Model Info")
    data = load_data()
    st.metric("Accuracy", "{:.1%}".format(model.score(scaler.transform(data.drop('Outcome', axis=1)), data['Outcome'])))
    st.info("Trained on Pima Indian Diabetes Dataset (768 patients)")
    st.markdown("[Dataset Source](https://www.kaggle.com/uciml/pima-indians-diabetes-database)")
