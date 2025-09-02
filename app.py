import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

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

        # Handle missing values
        imputer = SimpleImputer(strategy="mean")
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        # Train/test split
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
# App Logic (paste here ✅)
# -------------------------
st.title("🌾 Crop Production Prediction App")

df = load_data()
model = load_model()

st.subheader("Dataset Preview")
if df is not None and not df.empty:
    st.dataframe(df.head(20))   # 👈 shows first 20 rows of your dataset
else:
    st.error("No data found! Make sure 'data/cleaned_agriculture_data.csv' exists.")
    st.subheader("Make a Prediction")

# User Inputs
area = st.selectbox("Select Area (Country/Region):", df["Area"].unique())
crop = st.selectbox("Select Crop:", df["Item"].unique())
year = st.slider("Select Year:", int(df["Year"].min()), int(df["Year"].max()), step=1)

# Suggest default values from dataset
default_area_harvested = df[(df["Area"] == area) & (df["Item"] == crop)]["Area_harvested"].mean()
default_yield = df[(df["Area"] == area) & (df["Item"] == crop)]["Yield"].mean()

area_harvested = st.number_input("Enter Area Harvested (ha):", 
                                 value=float(default_area_harvested if pd.notnull(default_area_harvested) else 1000))
yield_val = st.number_input("Enter Yield (kg/ha):", 
                            value=float(default_yield if pd.notnull(default_yield) else 2000))

# Button to predict
if st.button("Predict Production"):
    input_df = pd.DataFrame({
        "Area": [area],
        "Item": [crop],
        "Year": [year],
        "Area_harvested": [area_harvested],
        "Yield": [yield_val]
    })
    
    # Encode categorical variables
    input_encoded = pd.get_dummies(input_df, columns=["Area", "Item"], drop_first=True)
    
    # Align columns with training features
    train_cols = pd.get_dummies(df, columns=["Area", "Item"], drop_first=True).drop("Production", axis=1).columns
    input_encoded = input_encoded.reindex(columns=train_cols, fill_value=0)
    
    # Predict
    prediction = model.predict(input_encoded)[0]
    st.success(f"🌱 Predicted Production: {prediction:,.0f} tons")