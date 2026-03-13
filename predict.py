import mlflow.pyfunc
import pandas as pd
import os

# Point to the folder containing the model file
LR_PATH = r"mlartifacts\1\models\m-0685a8cd675b4f21b25ceb5d353658b8\artifacts"
XGB_PATH = r"mlartifacts\1\models\m-dbaab589d8364ceba1bc9c83894e7403\artifacts"

# Load the future forecasting csv
future_df = pd.read_csv("data/future.csv")
X_forecast = future_df[["Speed", "Direction"]]

models_to_run = [
    {"name": "Linear_Regression", "path": LR_PATH},
    {"name": "XGBRegressor", "path": XGB_PATH}
]

# Get predictions for each model
for model_info in models_to_run:

    if os.path.exists(model_info["path"]):
        try:
            loaded_model = mlflow.pyfunc.load_model(model_info["path"])
            predictions = loaded_model.predict(X_forecast)

            predictions_df = future_df[["Speed"]].copy()
            predictions_df["Predicted_Power"] = predictions
            print(f"Top 5 predictions for {model_info["name"]}:\n{predictions_df.head()}")

        except Exception as e:
            print(f"Error loading model from path: {e}")
    else:

        print(f"Directory not found: {model_info['path']}")
