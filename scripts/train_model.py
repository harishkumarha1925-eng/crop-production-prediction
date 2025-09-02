import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
import pickle

def train_and_save_model(input_file, model_file, features_file):
    df = pd.read_csv(input_file)

    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=["Area","Item"], drop_first=True)

    # Features and target
    X = df_encoded.drop("Production", axis=1)
    y = df_encoded["Production"]

    # 🔹 Handle missing values
    imputer = SimpleImputer(strategy="mean")  # fill NaNs with column mean
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Save feature names
    os.makedirs(os.path.dirname(features_file), exist_ok=True)
    pd.DataFrame({"features": X.columns}).to_csv(features_file, index=False)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Models
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    }

    results = {}
    best_model = None
    best_score = -1

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        results[name] = {"R2": r2, "MAE": mae, "MSE": mse}

        print(f"{name} - R2: {r2:.4f}, MAE: {mae:.2f}, MSE: {mse:.2f}")

        if r2 > best_score:
            best_score = r2
            best_model = model

    # Save best model
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    with open(model_file, "wb") as f:
        pickle.dump(best_model, f)

    print(f"Best model saved to {model_file}")

if __name__ == "__main__":
    input_file = "data/cleaned_agriculture_data.csv"
    model_file = "models/crop_production_model.pkl"
    features_file = "models/model_features.csv"
    train_and_save_model(input_file, model_file, features_file)