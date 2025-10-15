import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import requests
import pickle
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh

# --- Auto-refresh dashboard every 60s ---
st_autorefresh(interval=60 * 1000, key="auto_refresh")

# --- Load trained model ---
brain = pickle.load(open('Weather.pkl', 'rb'))

# --- Page Config ---
st.set_page_config(
    page_title="🌫️PM2.5 Forecast Dashboard💨",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Background Gradient (Eco Green theme) ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #E8F5E9 0%, #C8E6C9 100%);
    }
    .css-18e3th9 {padding-top: 0rem;}
    </style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.markdown("""
    <div style="text-align:center; padding: 1rem; background-color:#A5D6A7; border-radius:10px;">
        <h1 style="color:#1B5E20;">🌫️ PM2.5 Forecast Dashboard 💨</h1>
        <h4 style="color:#2E7D32;">Real-time Air Quality & Next 48-Hour Predictions</h4>
        <p style="color:#388E3C;">Stay informed. Stay safe. Powered by Machine Learning 💡</p>
    </div>
""", unsafe_allow_html=True)

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

# --- Merge PM2.5 and weather ---
lat, lon = 13.0827, 80.2707  # Example: Chennai
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

# Fill any remaining NaNs with forward + backward fill
df.ffill(inplace=True)
df.bfill(inplace=True)

# --- Detect anomalies using IsolationForest ---
from sklearn.ensemble import IsolationForest
X = df[feature_cols]
# Initialize IsolationForest
iso = IsolationForest(
    n_estimators=100,      # number of trees
    contamination=0.05,    # fraction of expected anomalies (5% here)
    random_state=42
)
# Fit model
iso.fit(X)
# Predict anomalies
df['anomaly'] = iso.predict(X)  # -1 = anomaly, 1 = normal

# --- Last Observed PM2.5 ---
last_row = df.iloc[-1]

# --- Next Hour Forecast ---
X_input = pd.DataFrame([last_row[feature_cols]], columns=feature_cols)
# Reorder columns to match training order
X_input = X_input[brain.feature_names_in_]
next_hour_pred = brain.predict(X_input)[0]

# --- Metric Cards ---
col1, col2 = st.columns(2)
with col1:
    st.metric("🌤️ Last Observed PM2.5", f"{last_row['pm25']:.2f} µg/m³")
with col2:
    st.metric("🔮 Next Hour Forecast", f"{next_hour_pred:.2f} µg/m³")

# --- Air Quality Status ---
if next_hour_pred <= 50:
    st.success("🟢 Air Quality: Good")
elif next_hour_pred <= 100:
    st.info("🟡 Air Quality: Moderate")
elif next_hour_pred <= 150:
    st.warning("🟠 Air Quality: Unhealthy for Sensitive Groups")
else:
    st.error("🔴 Air Quality: Unhealthy")

# --- Recursive forecast for 24–48h ---
forecast_hours = st.slider("⏱️ Select Forecast Horizon (hours)", 1, 48, 24)
predictions = []
forecast_times = []
temp_row = last_row.copy()
for i in range(forecast_hours):
    X_input = pd.DataFrame([temp_row[feature_cols]], columns=feature_cols)
    # Reorder columns to match training order
    X_input = X_input[brain.feature_names_in_]
    y_pred = brain.predict(X_input)[0]
    predictions.append(y_pred)
    forecast_times.append(temp_row['utc_time'])

    # Update lags for next iteration
    temp_row['pm25_lag3'] = temp_row['pm25_lag2']
    temp_row['pm25_lag2'] = temp_row['pm25_lag1']
    temp_row['pm25_lag1'] = y_pred

    # Update rolling mean if used
    temp_row['pm25_roll3'] = np.mean([temp_row['pm25_lag1'], temp_row['pm25_lag2'], temp_row['pm25_lag3']])
    temp_row['pm25_roll6'] = (temp_row['pm25_roll6']*5 + y_pred)/6  # simple rolling update
    temp_row['pm25_roll12'] = (temp_row['pm25_roll12']*11 + y_pred)/12  # simple rolling update

    # Update utc_time
    temp_row['utc_time'] += pd.Timedelta(hours=1)

# --- Create Forecast DataFrame ---
forecast_df = pd.DataFrame({
    'utc_time': forecast_times,
    'pm25_pred': predictions
})
# --- Line chart: Historical + Forecast + Anomalies ---
st.subheader(f"{forecast_hours}-Hour PM2.5 Forecast")
plt.figure(figsize=(12,5))
plt.plot(df['utc_time'], df['pm25'], label='Actual PM2.5', color='blue')
plt.plot(forecast_df['utc_time'], forecast_df['pm25_pred'], label='Predicted PM2.5', color='orange')

# Highlight anomalies
anomalies = df[df['anomaly']==-1]
plt.scatter(anomalies['utc_time'], anomalies['pm25'], color='red', marker='x', label='Anomaly')

plt.xlabel("Time")
plt.ylabel("PM2.5 (µg/m³)")
plt.legend()
plt.grid(True)
st.pyplot(plt)

# --- Feature Importance ---
st.subheader("🌿 Feature Importance Ranking")
import numpy as np
importances = brain.feature_importances_
importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": np.round(importances, 3)
}).sort_values(by="Importance", ascending=False)
st.bar_chart(importance_df.set_index("Feature"))

# --- Footer ---
st.markdown("""
    <hr>
    <div style='text-align:center;color:#1B5E20;'>
        Made with ❤️ by <b>Suhail Sharif</b> | Powered by Streamlit & Open-Meteo API 🌍
    </div>
""", unsafe_allow_html=True)
