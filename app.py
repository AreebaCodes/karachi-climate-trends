import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Page setup
st.set_page_config(page_title="Karachi Climate Dashboard", page_icon="🌦️", layout="wide")

st.title("🌦️ Karachi Climate Change: 30-Year Trend & Forecast")
st.markdown("Analysis of Karachi's temperature and rainfall based on the NASA POWER dataset")

# Data load function (sirf ek baar, dynamic path ke saath)
@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), "karachi-data.csv")
    df = pd.read_csv(
        csv_path,
        skiprows=11,
        na_values=[-999, -999.0, -999.00]
    )
    df_complete = df[(df["YEAR"] >= 1996) & (df["YEAR"] <= 2025)]
    yearly = df_complete.groupby("YEAR").agg(
        Avg_Temp=("T2M", "mean"),
        Max_Temp=("T2M_MAX", "max"),
        Total_Rain=("PRECTOTCORR", "sum")
    ).reset_index()
    return yearly

yearly = load_data()

# Sidebar controls (Interactive Slider)
st.sidebar.header("⚙️ Dashboard Settings")
forecast_years_count = st.sidebar.slider("Kitne saal ka forecast dekhein?", min_value=1, max_value=15, value=5)

# Forecast Calculation
X = yearly[["YEAR"]]
future_years = pd.DataFrame({"YEAR": np.arange(2026, 2026 + forecast_years_count)})

temp_model = LinearRegression().fit(X, yearly["Avg_Temp"])
temp_trend = temp_model.predict(X)
temp_forecast = temp_model.predict(future_years)

rain_model = LinearRegression().fit(X, yearly["Total_Rain"])
rain_trend = rain_model.predict(X)
rain_forecast = rain_model.predict(future_years)

# Top Summary Cards (KPIs)
col1, col2, col3 = st.columns(3)
col1.metric("30-Year Avg Temp", f"{yearly['Avg_Temp'].mean():.1f} °C")
col2.metric("Highest Recorded Temp", f"{yearly['Max_Temp'].max():.1f} °C")
col3.metric("Avg Annual Rainfall", f"{yearly['Total_Rain'].mean():.0f} mm")

st.divider()

# Tabs for Clean UI
tab1, tab2, tab3 = st.tabs(["🌡️ Temperature Trend", "🌧️ Rainfall Trend", "📊 Forecast Data Table"])

with tab1:
    st.subheader(f"Temperature Trend & {forecast_years_count}-Year Forecast")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(yearly["YEAR"], yearly["Avg_Temp"], marker="o", color="#d9534f", label="Yearly Avg Temp (°C)")
    ax1.plot(yearly["YEAR"], temp_trend, linestyle="--", color="black", label="30-Year Trend")
    ax1.plot(future_years["YEAR"], temp_forecast, marker="s", linestyle=":", color="#f0ad4e", label="Forecast")
    ax1.set_ylabel("Temperature (°C)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    st.pyplot(fig1)

with tab2:
    st.subheader(f"Rainfall Trend & {forecast_years_count}-Year Forecast")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.bar(yearly["YEAR"], yearly["Total_Rain"], color="#5bc0de", alpha=0.8, label="Annual Rainfall (mm)")
    ax2.plot(yearly["YEAR"], rain_trend, linestyle="--", color="#0275d8", linewidth=2, label="30-Year Trend")
    ax2.plot(future_years["YEAR"], rain_forecast, marker="s", linestyle=":", color="#5cb85c", linewidth=2, label="Forecast")
    ax2.set_ylabel("Rainfall (mm)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    st.pyplot(fig2)

with tab3:
    st.subheader("Predicted Values")
    forecast_df = pd.DataFrame({
        "Year": future_years["YEAR"],
        "Predicted Avg Temp (°C)": np.round(temp_forecast, 2),
        "Predicted Rainfall (mm)": np.round(rain_forecast, 1)
    })
    st.dataframe(forecast_df, use_container_width=True)
