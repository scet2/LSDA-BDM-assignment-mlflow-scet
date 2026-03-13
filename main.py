import mlflow
import warnings
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from windrose import WindroseAxes

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

def read_csv_with_time_index(path):
    """Helper to read CSVs with a datetime index."""
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

#Plots
def create_plots(df):
    """
    Create exploratory data analysis plots for wind power data
    """
    fig = plt.figure(figsize=(25,4))

    # Speed vs Total (Power Curve nature)
    ax1 = fig.add_subplot(1,2,1)

    ax1.scatter(df["Speed"], df["Total"], color = 'yellowgreen', alpha=0.2)
    power_curve = df.groupby("Speed").median(numeric_only=True)["Total"]
    ax1.plot(power_curve.index, power_curve.values, "k:", label="Power Curve")
    ax1.set_title("Windspeed vs Power")
    ax1.set_ylabel("Power [MW]")
    ax1.set_xlabel("Windspeed [m/s]")
    ax1.legend()

    # Windrose Plot Preparation
    df_windrose = df.copy()
    df_windrose["Direction"] = df_windrose["Direction"].astype(str).str.strip().str.upper()

    # Compass mapping
    compass = {
        "N":0, "NNE":22.5, "NE":45, "ENE":67.5,
        "E":90, "ESE":112.5, "SE":135, "SSE":157.5,
        "S":180, "SSW":202.5, "SW":225, "WSW":247.5,
        "W":270, "WNW":292.5, "NW":315, "NNW":337.5
    }

    df_windrose["deg"] = df_windrose["Direction"].map(compass)
    df_windrose["deg"] = df_windrose["deg"].fillna(pd.to_numeric(df_windrose["Direction"], errors="coerce"))

    df_windrose["speed"] = pd.to_numeric(df_windrose["Speed"], errors="coerce")

    df_windrose = df_windrose.dropna(subset=["deg", "speed"])

    ax2 = WindroseAxes(fig, [0.55, 0.1, 0.35, 0.8])
    fig.add_axes(ax2)

    ax2.bar(
        df_windrose["deg"],
        df_windrose["speed"],
        normed=True,
        opening=0.8,
        edgecolor="white",
        cmap = plt.cm.Reds
    )

    ax2.set_legend(title="Wind Speed")
    ax2.set_title("Wind Speed vs Direction")

    return fig

mlflow.sklearn.autolog() 
mlflow.set_tracking_uri("http://127.0.0.1:5000") 

# Set the experiment and run name
experiment_name = "PowerForecasting" 
run_name = "XGB_vs_LR"  
register_name_lr = "LR_Model"
register_name_xgb = "XGB_Model"

mlflow.set_experiment(experiment_name)

# Set the Pipelines
preprocessor = ColumnTransformer(
    transformers=[
    ('Encoder', OneHotEncoder(handle_unknown='ignore'), ['Direction']),
    ('scale', StandardScaler(), ['Speed'])
])

pipelineLR = Pipeline([
    ('preprocess', preprocessor),
    ('model', LinearRegression())
])

pipelineXGB = Pipeline([
    ('preprocess', preprocessor),
    ('model', XGBRegressor(
        colsample_bytree=1.0,   
        learning_rate=0.1,      
        max_depth=3, 
        n_estimators=50, 
        subsample=0.8,
        random_state=42,
        objective='reg:squarederror' 
    ))
])

# Load the dataframes
power_df = read_csv_with_time_index("data/power.csv")
wind_df = read_csv_with_time_index("data/weather.csv")

# Merge the dataframes
merged = pd.merge_asof(power_df, wind_df, on='time', allow_exact_matches=True)
merged_df = merged.copy()
merged_df = merged_df.drop(columns=['ANM','Non-ANM','Lead_hours','Source_time'])
merged_df = merged_df.dropna()

# Extract X and y
X = merged_df[["Speed", "Direction"]]
y = merged_df["Total"]

# Start an MLflow run
with mlflow.start_run(run_name=run_name) as parent_run:

    # Save the plots
    os.makedirs("plots", exist_ok=True)
    eda_fig = create_plots(merged_df)
    eda_fig.savefig("plots/eda_plots.png")
    mlflow.log_artifact("plots/eda_plots.png")
    plt.close(eda_fig)

    models_to_run = [
        {"name": "Linear_Regression", "pipeline": pipelineLR, "reg_name": register_name_lr},
        {"name": "XGBRegressor", "pipeline": pipelineXGB, "reg_name": register_name_xgb}
    ]

    for model in models_to_run:
        # Nested run for the specific model type
        with mlflow.start_run(run_name=model["name"], nested=True):
            mse_scores = [] # Save for each split
            tscv = TimeSeriesSplit(n_splits=5)
            pipe = model["pipeline"] # Grab the pipeline

            for i, (train_index, test_index) in enumerate(tscv.split(X)):
                with mlflow.start_run(run_name=f"Split_{i}", nested=True):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    pipe.fit(X_train, y_train)
                    preds = pipe.predict(X_test)
                    mse = mean_squared_error(y_test, preds)
                
                    mse_scores.append(mse)

            # Compute and log the average MSE scores across splits
            avg_mse = np.mean(mse_scores)
            print(f'Avg MSE: {avg_mse}')
            mlflow.log_metric('Average MSE', avg_mse)

            # Final fit on all data for registering the model
            pipe.fit(X, y)
            mlflow.sklearn.log_model(pipe, "model", registered_model_name=model["reg_name"])

