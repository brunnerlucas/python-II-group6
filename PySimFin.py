# This script defines the PySimFin class, which allows us to interact with the SimFin API
# to retrieve financial data, perform predictions using XGBoost, and simulate a basic trading strategy.

# === Imports ===
import os
import logging
import joblib
import requests
import pandas as pd
import numpy as np

from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb


# === SimFin API Class ===
class PySimFin:
    def __init__(self):
        self.url = "https://backend.simfin.com/api/v3/"
        self.__load_dotenv()
        self.__token = os.getenv("API_KEY")

    def __load_dotenv(self):
        load_dotenv()

    def __create_headers(self):
        return {
            "Authorization": f"api-key {self.__token}",
            "Accept": "application/json"
        }

    def create_params_dictionary(self, ticker, start=None, end=None):
        params = {"ticker": ticker}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        return params

    def get_share_prices(self, ticker: str, start: str = None, end: str = None):
        url = self.url + "companies/prices/compact"
        headers = self.__create_headers()
        params = self.create_params_dictionary(ticker, start, end)

        logging.info(f"Requesting: {url} with params: {params}")
        response = requests.get(url, headers=headers, params=params)

        print(f"Full URL: {response.url}")
        print(f"Response Status: {response.status_code}")
        response.raise_for_status()

        data = response.json()
        
        # Prompt: Check if the API response has the expected structure (list with 'data' and 'columns'), 
        # then convert to DataFrame. If not, log a warning and return an empty DataFrame.
        if isinstance(data, list) and len(data) > 0 and "data" in data[0] and "columns" in data[0]:
            return pd.DataFrame(data[0]["data"], columns=data[0]["columns"])
        else:
            logging.warning(f"No data found or format is unexpected for {ticker}")
            return pd.DataFrame()

    def get_financial_statement(self, ticker: str, statements: str, period: str = "FY", start: str = None, end: str = None):
        url = self.url + "companies/statements/compact"
        headers = self.__create_headers()
        params = {"ticker": ticker, "statements": statements, "period": period}
        if start:
            params["start"] = start + "-01-01"
        if end:
            params["end"] = end + "-12-31"

        logging.info(f"Requesting Financial Statement: {url} with params: {params}")

        try:
            response = requests.get(url, headers=headers, params=params)
            print(f"Full URL: {response.url}")
            print(f"Response Status: {response.status_code}")
            response.raise_for_status()

            data = response.json()
            if isinstance(data, list) and data and "statements" in data[0] and data[0]["statements"]: 
                statement = data[0]["statements"][0]
                return pd.DataFrame(statement["data"], columns=statement["columns"])
            else:
                logging.warning(f"No financial statement data found for '{ticker}'.")
                return pd.DataFrame()

        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching financial statements: {e}")
            return pd.DataFrame()

    def get_general_data(self, ticker: str):
        url = self.url + "companies/general/compact"
        headers = self.__create_headers()
        params = {"ticker": ticker}

        logging.info(f"Requesting General Data: {url} with params: {params}")

        try:
            response = requests.get(url, headers=headers, params=params)
            print(f"Full URL: {response.url}")
            print(f"Response Status: {response.status_code}")
            response.raise_for_status()

            data = response.json()
            if data["data"]:
                return pd.DataFrame(data["data"], columns=data["columns"])
            else:
                logging.warning(f"No general data found for {ticker}.")
                return pd.DataFrame()

        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching general data: {e}")
            return pd.DataFrame()

    def companies(self): # Prompt: Return a list of all companies available in SimFin
        url = self.url + "companies/list"
        headers = self.__create_headers()

        logging.info(f"Requesting: {url}")
        response = requests.get(url, headers=headers)
        print(f"Full URL: {response.url}")
        print(f"Response Status: {response.status_code}")
        response.raise_for_status()

        data = response.json()
        return pd.DataFrame(data)

    def train_xgboost_model(self, data, last_date):
        model_dir = "models"
        Path(model_dir).mkdir(exist_ok=True)
        ticker_symbol = data['Ticker'].iloc[0]
        model_path = os.path.join(model_dir, f"{ticker_symbol}_latest_xgb.pkl")

        data.rename(columns={'Adjusted Closing Price': 'Close'}, inplace=True)
        data = data[['Close', 'Date']]
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index(['Date'])
        last_date = pd.to_datetime(last_date)
        data = data[(data.index > last_date - pd.DateOffset(days=100)) & (data.index <= last_date)]
        data = data.rename(columns={'Close': 'target'})

        df = data.copy()
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['dayofweek'] = df.index.dayofweek
        df['lag1'] = df['target'].shift(1)
        df['log_return'] = np.log(df['target'] / df['target'].shift(1))
        df['MA_10'] = df['target'].rolling(10).mean()
        df['MA_50'] = df['target'].rolling(50).mean()
        df['Volatility'] = df['target'].rolling(10).std()

        def compute_rsi(series, window=14): # Prompt: Calculate Relative Strength Index (RSI) as part of feature engineering
            delta = series.diff()
            gain = delta.where(delta > 0, 0).rolling(window).mean()
            loss = -delta.where(delta < 0, 0).rolling(window).mean()
            rs = gain / (loss + 1e-10)
            return 100 - (100 / (1 + rs))

        df['RSI'] = compute_rsi(df['target'])
        df['BB_Upper'] = df['MA_10'] + (2 * df['Volatility'])
        df['BB_Lower'] = df['MA_10'] - (2 * df['Volatility'])
        df.dropna(inplace=True)

        if len(df) < 10:
            raise ValueError("Not enough data after preprocessing to train or predict.")

        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else: # Prompt: If no saved model exists, train a new XGBoost regressor and save it
            X = df.drop(columns=['target'])
            y = df['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
            model.fit(X_train, y_train)
            joblib.dump(model, model_path)

        future_date = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=1, freq='B')
        future_df = pd.DataFrame(index=pd.MultiIndex.from_product([future_date], names=['Date']))
        last_row = df.iloc[-1]

        for col in ['year', 'month', 'day', 'dayofweek', 'lag1', 'log_return', 'MA_10', 'MA_50',
                    'Volatility', 'RSI', 'BB_Upper', 'BB_Lower']:
            future_df[col] = getattr(last_row, col)

        pred = model.predict(future_df)
        return pd.DataFrame({'Date': future_date, 'Predicted_Close': pred})

# Prompt: Simulate a hybrid trading strategy that mixes holding and prediction-based trading (but further adjusted)
    def simulate_hybrid_hold_strategy(self, df, model_func, ticker, initial_cash=10000,
                                      buy_threshold=0.005, sell_threshold=0.03, trade_fraction=0.5):
        cash, shares = initial_cash, 0
        equity_curve, trade_log = [], []

        df = df.copy()
        df = df.sort_values("Date").reset_index(drop=True)
        df['Date'] = pd.to_datetime(df['Date'])

        for i in range(60, len(df) - 1):
            today = df.loc[i, 'Date']
            tomorrow = df.loc[i + 1, 'Date']
            today_close = df.loc[i, 'Close']
            tomorrow_close = df.loc[i + 1, 'Close']
            train_data = df.iloc[i - 60:i + 1].copy()

            try:
                prediction_df = model_func(train_data, today)
                predicted_close = prediction_df['Predicted_Close'].values[0]
            except Exception as e:
                continue

            pct_diff = (predicted_close - today_close) / today_close
            action = None

            if pct_diff > buy_threshold:
                buy_amount = cash * trade_fraction
                qty = int(buy_amount // today_close)
                if qty > 0:
                    cost = qty * today_close
                    cash -= cost
                    shares += qty
                    action = f"Bought {qty} shares at ${today_close:.2f} on {today.date()} (predicted ${predicted_close:.2f})"

            elif (tomorrow_close - today_close) / today_close < -sell_threshold:
                qty = int(shares * trade_fraction)
                if qty > 0:
                    revenue = qty * tomorrow_close
                    cash += revenue
                    shares -= qty
                    action = f"Sold {qty} shares at ${tomorrow_close:.2f} on {tomorrow.date()} (drop after ${today_close:.2f})"

            total_value = cash + (shares * tomorrow_close)
            equity_curve.append({'Date': tomorrow, 'Capital': total_value})
            if action:
                trade_log.append(action)

        return pd.DataFrame(equity_curve), trade_log


# === Main ===
if __name__ == "__main__":
    simfin_api = PySimFin()
    df_companies = simfin_api.companies()
    print(df_companies.head())

# This code has been refactored with the assistance of ChatGPT to enhance structure,
# modularity, and adherence to clean coding principles.