import pandas as pd
import os
from datetime import datetime, timedelta, timezone
import requests
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import time

# --- Feature columns ---
feature_cols = [
    'pm25_lag1','pm25_lag2','pm25_lag3',
    'pm25_roll3','pm25_roll6','pm25_roll12',
    'temperature_2m','relativehumidity_2m',
    'windspeed_10m','winddirection_10m','surface_pressure'
    ]

# === Function to extract datetime and PM2.5 from Open-Meteo Air Quality ===
def fetch_openmeteo_pm25(lat, lon):
    """Fetch PM2.5 for the past 72 hours (3 days) from Open-Meteo Air Quality API."""
    # Current UTC date
    end_date = datetime.now(timezone.utc)
    # Start date = Last 72 hours
    start_date = end_date - timedelta(hours=72)

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm2_5",
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "timezone": "UTC"
    }

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    js = r.json()["hourly"]

    df = pd.DataFrame({
        "utc_time": pd.to_datetime(js["time"], utc=True),
        "pm25": js["pm2_5"]
    })
    # Keep only last 72 hours
    df = df[(df["utc_time"] >= start_date) & (df["utc_time"] <= end_date)]
    return df

# === Function to extract datetime and required variables from Open-Meteo Forecast ===
def fetch_openmeteo_weather(lat, lon):
    """Fetch hourly weather variables from Open-Meteo."""
    # Current UTC date
    end_date = datetime.now(timezone.utc)
    # Start date = Last 72 hours
    start_date = end_date - timedelta(hours=72)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m,winddirection_10m,surface_pressure",
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "timezone": "UTC"
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    hourly = r.json()["hourly"]
    df = pd.DataFrame({"utc_time": pd.to_datetime(hourly["time"], utc=True)})
    for k, v in hourly.items():
        if k != "time":
            df[k] = v
    df = df[(df["utc_time"] >= start_date) & (df["utc_time"] <= end_date)]
    return df

def retrain():
    # --- Merge PM2.5 and weather ---
    lat, lon = 13.0827, 80.2707  # Chennai
    pm25_df = fetch_openmeteo_pm25(lat, lon)
    weather_df = fetch_openmeteo_weather(lat, lon)
    df = pd.merge(pm25_df, weather_df, on="utc_time", how="inner")
    # --- Feature Engineering: Lags & Rolling Means ---
    pm_lags = [1,2,3]  # choose lags depending on forecast horizon
    for lag in pm_lags:
      df[f"pm25_lag{lag}"] = df["pm25"].shift(lag)
    df['pm25_roll3'] = df['pm25'].rolling(window=3).mean()
    df['pm25_roll6'] = df['pm25'].rolling(window=6).mean()
    df['pm25_roll12'] = df['pm25'].rolling(window=12).mean()
    # --- Ensured target is shifted properly to predict next hour’s PM2.5 ---
    df['target_pm25'] = df['pm25'].shift(-1)
    # --- Fill any remaining NaNs with forward + backward fill ---
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    # --- Defined Features and Target columns ---
    # Target variable
    y = df['target_pm25']
    # Feature variables (exclude utc_time and pm25 itself)
    X = df.drop(columns=['utc_time', 'pm25', 'target_pm25'], errors='ignore')
    # --- Splited data into Train and Test (train_test_split) ---
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42) # Used the first 80% for training and the last 20% for testing.
    # --- Retrain model ---
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    # --- Save model safely ---
    with open("Weather_temp.pkl", "wb") as f:
        pickle.dump(model, f)
    os.replace("Weather_temp.pkl", "Weather.pkl")
    print(f"Model retrained at {datetime.now()}")

# --- Run once ---
retrain()
