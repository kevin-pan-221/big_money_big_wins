import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained XGBoost model and scaler
xgb_model = joblib.load("models/xgb_model.pkl")  # Replace with the correct path if needed
#scaler = joblib.load("models/scaler.pkl")  # Replace with the correct path if needed

# Title and description
st.title("NFL Wins Prediction App")
st.markdown("""
This app predicts the number of wins based on team salary distribution across different positions. 
You can adjust individual salary percentages using sliders or select predefined scenarios to see how the model responds.
""")

# Sidebar for input features
st.sidebar.header("Input Features")

# User-defined or Predefined Scenarios
option = st.sidebar.selectbox(
    "Choose Input Method",
    ["Custom Input", "Balanced Offense/Defense", "High QB Focus", "High Defense Focus"]
)

if option == "Custom Input":
    # Custom input using sliders
    QB_P = st.sidebar.slider("QB_P (Quarterback %)", 0.0, 100.0, 15.0, step=1.0)
    RB_P = st.sidebar.slider("RB_P (Running Back %)", 0.0, 100.0, 8.0, step=1.0)
    WR_P = st.sidebar.slider("WR_P (Wide Receiver %)", 0.0, 100.0, 20.0, step=1.0)
    TE_P = st.sidebar.slider("TE_P (Tight End %)", 0.0, 100.0, 10.0, step=1.0)
    OL_P = st.sidebar.slider("OL_P (Offensive Line %)", 0.0, 100.0, 25.0, step=1.0)
    IDL_P = st.sidebar.slider("IDL_P (Interior Defensive Line %)", 0.0, 100.0, 12.0, step=1.0)
    EDGE_P = st.sidebar.slider("EDGE_P (Edge %)", 0.0, 100.0, 18.0, step=1.0)
    LB_P = st.sidebar.slider("LB_P (Linebacker %)", 0.0, 100.0, 9.0, step=1.0)
    S_P = st.sidebar.slider("S_P (Safety %)", 0.0, 100.0, 13.0, step=1.0)
    CB_P = st.sidebar.slider("CB_P (Cornerback %)", 0.0, 100.0, 11.0, step=1.0)

    # Collect custom inputs
    raw_inputs = np.array([QB_P, RB_P, WR_P, TE_P, OL_P, IDL_P, EDGE_P, LB_P, S_P, CB_P])
else:
    # Predefined scenarios
    if option == "Balanced Offense/Defense":
        raw_inputs = np.array([15, 15, 15, 10, 10, 10, 10, 5, 5, 5, 50, 50])
    elif option == "High QB Focus":
        raw_inputs = np.array([40, 5, 10, 5, 10, 5, 5, 5, 5, 5, 25, 35])
    elif option == "High Defense Focus":
        raw_inputs = np.array([5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 75, 25])

# Normalize to ensure percentages sum to 100
total = raw_inputs.sum()
if total != 100:
    normalized_inputs = (raw_inputs / total) * 100
    st.sidebar.warning("The input percentages have been normalized to sum to 100%.")
else:
    normalized_inputs = raw_inputs

# Display normalized inputs
input_data = pd.DataFrame([normalized_inputs], columns=[
    "QB_P", "RB_P", "WR_P", "TE_P", "OL_P", "IDL_P", 
    "EDGE_P", "LB_P", "S_P", "CB_P"
])
st.subheader("Normalized Input Features")
st.write(input_data)

# Scale the normalized input data
#scaled_data = scaler.transform(input_data)

# Make a prediction
#prediction = xgb_model.predict(scaled_data)
prediction = xgb_model.predict(input_data)

# Display the prediction
st.subheader("Prediction")
st.write(f"Predicted Wins: **{prediction[0]:.2f}**")

# Add Feature Importance Visualization
st.subheader("Feature Importance")
feature_importances = xgb_model.feature_importances_
plt_df = pd.DataFrame({
    "Feature": input_data.columns,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

# Display feature importances in a bar chart
st.bar_chart(plt_df.set_index("Feature"))