import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

st.title("📈 Sales Forecasting Dashboard")
st.caption("Real-world retail demand forecasting using ARIMA")

DATA_PATH = "../data/store1_item1_sales.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    # 🔥 FIX 1: Robust date parsing (handles ALL formats safely)
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
        dayfirst=True
    )

    # Drop rows where date could not be parsed
    df = df.dropna(subset=["date"])

    # 🔥 FIX 2: AGGREGATE BEFORE MODELING (CRITICAL)
    daily_sales = (
        df.groupby("date", as_index=False)["sales"]
        .sum()
        .sort_values("date")
    )

    daily_sales.set_index("date", inplace=True)
    daily_sales = daily_sales.asfreq("D")

    return daily_sales


# ======================
# Load data
# ======================
ts = load_data()

st.subheader("📊 Historical Daily Sales")
st.line_chart(ts)

# ======================
# Forecast settings
# ======================
forecast_days = st.slider("Select forecast horizon (days)", 7, 60, 30)

# ======================
# Train ARIMA
# ======================
with st.spinner("Training ARIMA model..."):
    model = ARIMA(ts, order=(5, 1, 0))
    model_fit = model.fit()

forecast = model_fit.forecast(steps=forecast_days)

forecast_df = forecast.to_frame(name="forecast_sales")

# ======================
# Plot forecast
# ======================
st.subheader("🔮 Sales Forecast")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(ts.index, ts["sales"], label="Historical Sales")
ax.plot(forecast_df.index, forecast_df["forecast_sales"], label="Forecast", linestyle="--")
ax.legend()
ax.set_ylabel("Sales")
ax.set_xlabel("Date")

st.pyplot(fig)

# ======================
# Forecast table
# ======================
st.subheader("📄 Forecast Data")
st.dataframe(forecast_df.reset_index())

# ======================
# Save forecast
# ======================
forecast_df.reset_index().to_csv("../data/future_forecast.csv", index=False)
st.success("Forecast saved to data/future_forecast.csv")
