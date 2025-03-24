import os
import pandas as pd
from dotenv import load_dotenv
import requests

import requests
import os
import logging
import pandas as pd
from dotenv import load_dotenv

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
        if isinstance(data, list) and len(data) > 0 and "data" in data[0] and "columns" in data[0]:
            df = pd.DataFrame(data[0]["data"], columns=data[0]["columns"])
            return df
        else:
            logging.warning(f"No data found or format is unexpected for {ticker}")
            return pd.DataFrame()

            
    def get_financial_statement(self, ticker: str, statements: str, period: str= "FY", fyear=None):
        url = self.url + "companies/statements/compact"
        headers = self.__create_headers()
        params = {
            "ticker": ticker,
            "statements": statements,
            "period": period
        }      
        
        if fyear:
            params["fyear"]=fyear
    
        logging.info(f"Requesting Financial Statement: {url} with params: {params}")
    
        try:
            response = requests.get(url, headers=headers, params=params)
            print(f"Full URL: {response.url}")
            print(f"Response Status: {response.status_code}")
            response.raise_for_status()
    
            data = response.json()
            if isinstance(data, list) and "statements" in data[0]:
                statement = data[0]["statements"][0]
                return pd.DataFrame(statement["data"], columns=statement["columns"])
            else:
                logging.warning(f"No financial statement data found for {ticker}.")
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



# Example usage
if __name__ == "__main__":
    simfin_api = PySimFin()

    #df_prices = simfin_api.get_share_prices("AAPL", "2024-12-01", "2024-12-31") 
    #print(df_prices)

    df_financials = simfin_api.get_financial_statement("AAPL","Derived","FY","2022")
    #print(df_financials.head())

    #df_general=simfin_api.get_general_data("AAPL")
