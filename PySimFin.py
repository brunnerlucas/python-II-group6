import os
import pandas as pd
from dotenv import load_dotenv
import requests

import requests
import os
import logging
import pandas as pd
from dotenv import load_dotenv



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

class PySimFin:
    def __init__(self):
        self.url = "https://backend.simfin.com/api/v3/" 
        self.__load_dotenv()
        self.__token = "122148ca-24f4-4ee5-a10e-4d347614ea86"

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
        if isinstance(data, list) and len(data) > 0 and "data" in data[0] and "columns" in data[0]:
            df = pd.DataFrame(data[0]["data"], columns=data[0]["columns"])
            return df
        else:
            logging.warning(f"No data found or format is unexpected for {ticker}")
            return pd.DataFrame()


    def get_financial_statement(self, ticker: str, statements: str, period: str = "FY", start: str = None, end: str = None):
        url = self.url + "companies/statements/compact"
        headers = self.__create_headers()
        
        params = {
            "ticker": ticker,
            "statements": statements,
            "period": period
        }
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
            if isinstance(data, list) and len(data) > 0 and "statements" in data[0] and len(data[0]["statements"]) > 0:
                statement = data[0]["statements"][0]
                return pd.DataFrame(statement["data"], columns=statement["columns"])
            else:
                logging.warning(f"No financial statement data found or available for '{ticker}'.")
                return pd.DataFrame()
    
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching financial statements: {e}")
            return pd.DataFrame()


    def get_general_data(self, ticker: str):
        url = self.url + "companies/general/compact"
        headers = self.__create_headers()
        params = {
            "ticker": ticker,
        }      
        
        logging.info(f"Requesting Financial Statement: {url} with params: {params}")
    
        try:
            response = requests.get(url, headers=headers, params=params)
            print(f"Full URL: {response.url}")
            print(f"Response Status: {response.status_code}")
            response.raise_for_status()
    
            data = response.json()
            if len(data["data"])>0:
                #general_data = data["data"][0]
                return pd.DataFrame(data["data"], columns=data["columns"])
            else:
                logging.warning(f"No financial statement data found for {ticker}.")
                return pd.DataFrame()       
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching financial statements: {e}")
            return pd.DataFrame()
        
    def companies(self):
        url = self.url + "companies/list"
        headers = self.__create_headers()
        params = {}
    
        logging.info(f"Requesting: {url} with params: {params}")
    
        response = requests.get(url, headers=headers, params=params)
        print(f"Full URL: {response.url}")
        print(f"Response Status: {response.status_code}")
    
        response.raise_for_status()
    
        data = response.json()
        df = pd.DataFrame(data)
        #df = pd.DataFrame(data[0]["data"], columns=data[0]["columns"])
        """if isinstance(data, list) and len(data) > 0 and "data" in data[0] and "columns" in data[0]:
            df = pd.DataFrame(data[0]["data"], columns=data[0]["columns"])
            return df
        else:
            logging.warning(f"No data found or format is unexpected for companies.")"""
        return df
    


    def train_xgboost_model(self, data, last_date):
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
    

    def simulate_hybrid_hold_strategy(self, df, model_func, ticker, initial_cash=10000,
                                    buy_threshold=0.005, sell_threshold=0.03,
                                    trade_fraction=0.5):
        cash = initial_cash
        shares = 0
        equity_curve = []
        trade_log = []

        df = df.copy()
        df = df.sort_values("Date").reset_index(drop=True)
        df['Date'] = pd.to_datetime(df['Date'])

        for i in range(60, len(df) - 1):  # start after 60 days
            today = df.loc[i, 'Date']
            tomorrow = df.loc[i + 1, 'Date']
            today_close = df.loc[i, 'Close']
            tomorrow_close = df.loc[i + 1, 'Close']

            train_data = df.iloc[i-60:i+1].copy()

            try:
                prediction_df = model_func(train_data, today)
                predicted_close = prediction_df['Predicted_Close'].values[0]
            except:
                continue

            pct_diff = (predicted_close - today_close) / today_close
            action = None

            # 🟢 Buy condition
            if pct_diff > buy_threshold:
                buy_amount = cash * trade_fraction
                qty = int(buy_amount // today_close)
                cost = qty * today_close
                if qty > 0:
                    cash -= cost
                    shares += qty
                    action = f"Bought {qty} shares at ${today_close:.2f} on {today.date()} (predicted ${predicted_close:.2f})"

            # 🔴 Sell condition
            elif (tomorrow_close - today_close) / today_close < -sell_threshold:
                qty = int(shares * trade_fraction)
                revenue = qty * tomorrow_close
                if qty > 0:
                    cash += revenue
                    shares -= qty
                    action = f"Sold {qty} shares at ${tomorrow_close:.2f} on {tomorrow.date()} (drop after ${today_close:.2f})"

            # Track capital
            total_value = cash + (shares * tomorrow_close)
            equity_curve.append({'Date': tomorrow, 'Capital': total_value})

            # Add to log
            if action:
                trade_log.append(action)

        return pd.DataFrame(equity_curve), trade_log




# Example usage
if __name__ == "__main__":
    simfin_api = PySimFin()

    #df_prices = simfin_api.get_share_prices("AAPL", "2024-12-01", "2024-12-31") 
    #print(df_prices)

    df_financials = simfin_api.companies()
    print(df_financials.head())
    #print(df_financials.head())

    #df_general=simfin_api.get_general_data("AAPL")
