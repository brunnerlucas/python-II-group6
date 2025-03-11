import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

st.set_page_config(layout="wide", page_title="AI Stock Trading System", page_icon="📈")


@st.cache_data
def read_and_preprocess_data():
    key = "us_shareproce_joined_companies"
    data = pd.read_csv(f"data/enrich/{key}.csv")
    
    # Convert Date column to datetime format
    data['Date'] = pd.to_datetime(data['Date'])

    # Filter for Technology sector stocks with high volume
    data = data[(data['Sector'] == 'Technology') & (data['Volume'] > 100000000)]
    selected_tickers = ['NVDA', 'BB', 'SNAP', 'HYSR', 'AMD', 'META']
    data_filtered = data[data['Ticker'].isin(selected_tickers)]
    # Ensure data is sorted by date
    data_filtered = data_filtered.sort_values(by=['Ticker', 'Date'])

# Create binary target: 1 if next day's Close is higher, else 0
    data_filtered['Target'] = 0  # Default: Hold
    data_filtered.loc[data_filtered['Close'].shift(-1) > data_filtered['Close'] * 1.02, 'Target'] = 1  # Buy if price up >2%
    data_filtered.loc[data_filtered['Close'].shift(-1) < data_filtered['Close'] * 0.98, 'Target'] = 2  # Sell if price down >2%

    data_filtered.dropna(inplace=True)

# Drop NaN values created by shift
    data_filtered.dropna(inplace=True)

        # Moving Averages (Trend Indicators)
    data_filtered['MA_10'] = data_filtered.groupby('Ticker')['Close'].transform(lambda x: x.rolling(10).mean())
    data_filtered['MA_50'] = data_filtered.groupby('Ticker')['Close'].transform(lambda x: x.rolling(50).mean())

    # Volatility Indicator
    data_filtered['Volatility'] = data_filtered.groupby('Ticker')['Close'].transform(lambda x: x.rolling(10).std())

    # Relative Strength Index (Momentum Indicator)
    def compute_rsi(series, window=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / (loss + 1e-10)  # Avoid division by zero
        return 100 - (100 / (1 + rs))

    data_filtered['RSI'] = data_filtered.groupby('Ticker')['Close'].transform(lambda x: compute_rsi(x))

    # Bollinger Bands (Market Volatility)
    data_filtered['BB_Upper'] = data_filtered['MA_10'] + (2 * data_filtered['Volatility'])
    data_filtered['BB_Lower'] = data_filtered['MA_10'] - (2 * data_filtered['Volatility'])

    data_filtered.dropna(inplace=True)

    return data_filtered

#def get_realtime_data():

def main():

    # Load stock data
    data = read_and_preprocess_data()
    tickers = ['NVDA', 'BB', 'SNAP', 'HYSR', 'AMD', 'META']

    # Load trained ML model
    ml_model = joblib.load("ml_model.pkl")

    # Define ML Model Features (must match training features)
    features = ['Open', 'High', 'Low', 'Volume','MA_10','MA_50', 'Volatility', 'RSI', 'BB_Upper', 'BB_Lower']
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Go Live", 'Trading Strategy'])

    # Home Page
    if page == "Home":
        st.title("Stock Trading System")
        st.subheader("Overview")


    # Go Live Page
    elif page == "Go Live":
        st.title("AI Trading System")

        # Sidebar Inputs
        st.sidebar.header("Stock Analysis Options")
        ticker = st.sidebar.selectbox("Select Stock Ticker", tickers)
        start_date = st.sidebar.date_input("Start Date", datetime(2023, 1, 1))
        end_date = st.sidebar.date_input("End Date", datetime(2024, 1, 1))

        # Filter Data for Selected Stock
        stock_data = data[(data['Ticker'] == ticker) & (data['Date'] >= pd.to_datetime(start_date)) & (data['Date'] <= pd.to_datetime(end_date))]

        if stock_data.empty:
            st.warning("No data available for the selected ticker and date range.")
        
        else:
            st.subheader(f"📊 {ticker} Stock Prices")
            # Plot stock prices
            plt.figure(figsize=(10, 5))
            plt.plot(stock_data["Date"], stock_data["Close"], label="Closing Price", color='blue')
            plt.xlabel("Date")
            plt.ylabel("Price (USD)")
            plt.title(f"{ticker} Stock Price Trend")
            plt.legend()
            st.pyplot(plt)

        # ML Model Prediction Button
        if st.sidebar.button("Predict Buy/Hold/Sell"):
            if not stock_data.empty:
                # Extract latest available stock data
                latest_data = stock_data[features].iloc[-1].values.reshape(1, -1)

                # Predict Buy (1), Hold (0), or Sell (2)
                prediction = ml_model.predict(latest_data)[0]
                actions = {0: "HOLD", 1: "BUY", 2: "SELL"}
                decision = actions[prediction]
                st.subheader(f"ML Model Prediction for {ticker}: {decision}")


if __name__ == "__main__":
    st.title("AI Stock Trading System")
    main()
