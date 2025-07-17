import streamlit as st
import numpy as np
import joblib

# Load scaler and model
scaler = joblib.load("Scaler.pkl")
model = joblib.load("model.pkl")

# Set page config
st.set_page_config(page_title="Real Estate Price Predictor 🏡", page_icon=":house:", layout="centered")

# Main title
st.markdown("<h1 style='text-align: center; color: #4B9CD3;'>🏠 Real Estate Price Prediction App</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar for inputs
st.sidebar.header("📝 Enter Property Details")

bed = st.sidebar.number_input("🛏️ Bedrooms", min_value=0, value=2, step=1)
bath = st.sidebar.number_input("🛁 Bathrooms", min_value=0, value=1, step=1)
size = st.sidebar.number_input("📐 Size (in sqft)", min_value=100, value=1000, step=50)

st.sidebar.markdown("---")
predictbutton = st.sidebar.button("🚀 Predict Price")

# Display inputs back to user
col1, col2, col3 = st.columns(3)
col1.metric("Bedrooms", bed)
col2.metric("Bathrooms", bath)
col3.metric("Size (sqft)", size)

st.markdown("### 🔍 Prediction Result")
st.markdown("---")

# Prediction logic
if predictbutton:
    X = np.array([bed, bath, size])
    X_scaled = scaler.transform([X])
    prediction = model.predict(X_scaled)[0]

    st.balloons()
    st.success(f"🎉 The estimated property price is **₹ {prediction:,.2f}**")
    st.markdown("💡 *Note: This is a machine learning-based estimate and may vary from actual market prices.*")

else:
    st.info("👈 Fill out the property details on the left and click **Predict Price** to get an estimate.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Created with ❤️ by Rohit Kumar</div>", unsafe_allow_html=True)
