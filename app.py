import streamlit as st
import pandas as pd
import pickle

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned_agriculture_data.csv")  # fixed path
    return df

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    with open("models/crop_production_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# -------------------------
# Load Features
# -------------------------
def load_features():
    return pd.read_csv("models/model_features.csv")["features"].tolist()

# -------------------------
# App Start
# -------------------------
st.title("🌾 Crop Production Prediction App")
st.write("Predict agricultural production (tons) based on area harvested, yield, crop, and year.")

# Load resources
df = load_data()
model = load_model()
train_cols = load_features()

# -------------------------
# User Inputs
# -------------------------
area = st.selectbox("Select Area (Country/Region):", df["Area"].unique())
crop = st.selectbox("Select Crop:", df["Item"].unique())
year = st.slider("Select Year:", int(df["Year"].min()), int(df["Year"].max()), step=1)

# Default input values
default_area_harvested = df[(df["Area"] == area) & (df["Item"] == crop)]["Area_harvested"].mean()
default_yield = df[(df["Area"] == area) & (df["Item"] == crop)]["Yield"].mean()

area_harvested = st.number_input("Enter Area Harvested (ha):", 
                                 value=float(default_area_harvested if pd.notnull(default_area_harvested) else 1000))
yield_val = st.number_input("Enter Yield (kg/ha):", 
                            value=float(default_yield if pd.notnull(default_yield) else 2000))

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Production"):
    input_df = pd.DataFrame({
        "Area": [area],
        "Item": [crop],
        "Year": [year],
        "Area_harvested": [area_harvested],
        "Yield": [yield_val]
    })
    
    # Encode input same as training
    input_encoded = pd.get_dummies(input_df, columns=["Area", "Item"], drop_first=True)
    
    # Align columns with training set
    input_encoded = input_encoded.reindex(columns=train_cols, fill_value=0)
    
    # Predict
    prediction = model.predict(input_encoded)[0]
    st.success(f"🌱 Predicted Production: {prediction:,.0f} tons")

# -------------------------
# Show Data Sample
# -------------------------
if st.checkbox("Show Sample Data"):
    st.dataframe(df.sample(10))