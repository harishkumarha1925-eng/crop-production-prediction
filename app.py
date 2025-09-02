import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned_agriculture_data.csv")  
    return df

# -------------------------
# Load or Train Model
# -------------------------
@st.cache_resource
def load_model():
    model_path = "models/crop_production_model.pkl"
    if not os.path.exists(model_path):
        # Train a simple model on the fly
        df = pd.read_csv("data/cleaned_agriculture_data.csv")
        df_encoded = pd.get_dummies(df, columns=["Area", "Item"], drop_first=True)
        X = df_encoded.drop("Production", axis=1)
        y = df_encoded["Production"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        os.makedirs("models", exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    else:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    return model

# -------------------------
# App Logic (UI)
# -------------------------
st.title("🌾 Crop Production Prediction App")

df = load_data()
model = load_model()