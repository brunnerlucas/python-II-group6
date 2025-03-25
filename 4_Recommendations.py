import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np

def train_xgboost_model(data, last_date):
        # Filter the DataFrame to get the last 60 days of data
        data.rename(columns={'Adjusted Closing Price': 'Close'}, inplace=True)
        data = data[['Close','Date']]
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index(['Date'])
        last_date = pd.to_datetime(last_date)
        sixty_days_prior = last_date - pd.DateOffset(days=60)
        data = data[(data.index.get_level_values('Date') > sixty_days_prior) & (data.index.get_level_values('Date') <= last_date)]

        # Rename close to target
        data = data.rename(columns={'Close': 'target'})

        data_filtered = data.copy()

        
        # Extract useful features from the Date column
        data_filtered['year'] = data_filtered.index.get_level_values('Date').year
        data_filtered['month'] = data_filtered.index.get_level_values('Date').month
        data_filtered['day'] = data_filtered.index.get_level_values('Date').day
        data_filtered['dayofweek'] = data_filtered.index.get_level_values('Date').dayofweek
        data_filtered['lag1'] = data_filtered['target'].shift(1)
        data_filtered['log_return'] = np.log(data_filtered['target'] / data_filtered['target'].shift(1))

        # Moving Averages (Trend Indicators)
        data_filtered['MA_10'] = data_filtered['target'].rolling(10).mean()
        data_filtered['MA_50'] = data_filtered['target'].rolling(50).mean()

        # Volatility Indicator
        data_filtered['Volatility'] = data['target'].rolling(10).std()

        # Relative Strength Index (Momentum Indicator)
        def compute_rsi(series, window=14):
            delta = series.diff()
            gain = delta.where(delta > 0, 0).rolling(window=window).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
            rs = gain / (loss + 1e-10)  # Avoid division by zero
            return 100 - (100 / (1 + rs))

        data_filtered['RSI'] = compute_rsi(data_filtered['target'])

        # Bollinger Bands (Market Volatility)
        data_filtered['BB_Upper'] = data_filtered['MA_10'] + (2 * data_filtered['Volatility'])
        data_filtered['BB_Lower'] = data_filtered['MA_10'] - (2 * data_filtered['Volatility'])

        #data_filtered.dropna(inplace=True)




        # Prepare the data
        X = data_filtered.drop(columns=['target'])
        y = data_filtered['target']
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train an XGBoost model
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Evaluate the model
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        # Predict the next 15 days
        future_dates = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=1, freq='B')  # Business days
        future_data = pd.DataFrame(index=pd.MultiIndex.from_product([ future_dates], names=[ 'Date']))
        future_data['year'] = future_data.index.get_level_values('Date').year
        future_data['month'] = future_data.index.get_level_values('Date').month
        future_data['day'] = future_data.index.get_level_values('Date').day
        future_data['dayofweek'] = future_data.index.get_level_values('Date').dayofweek
        future_data['lag1'] = data_filtered['target'].iloc[-1]
        future_data['log_return'] = data_filtered['log_return'].iloc[-1]
        future_data['MA_10'] = data_filtered['MA_10'].iloc[-1]
        future_data['MA_50'] = data_filtered['MA_50'].iloc[-1]
        future_data['Volatility'] = data_filtered['Volatility'].iloc[-1]
        future_data['RSI'] = data_filtered['RSI'].iloc[-1]
        future_data['BB_Upper'] = data_filtered['BB_Upper'].iloc[-1]
        future_data['BB_Lower'] = data_filtered['BB_Lower'].iloc[-1]

        
        
        future_predictions = model.predict(future_data)
        future_predictions_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_predictions})
        
        return future_predictions_df










import streamlit as st
from PySimFin import PySimFin

#st.title("General Company Information")

api = PySimFin()

#ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", value="AAPL")

ticker = st.text_input("Stock Ticker")

if st.button("Get Company Info"):

#if st.button("Get Company Info"):
    df = api.get_share_prices(ticker, start="2025-01-01", end="2025-03-24")

    previous_price = df[df['Date'] == '2025-03-21'][['Date','Last Closing Price']]

    real_price = df[df['Date'] == '2025-03-24'][['Date','Last Closing Price']]

    st.write(previous_price)

    st.write(real_price)


    price_predicted = train_xgboost_model(df, '2025-03-23')

    st.write(price_predicted.head())

else:
    st.warning("No company information found.")







