import streamlit as st
from PySimFin import PySimFin

api = PySimFin()

st.set_page_config(page_title="SimFin Stock Explorer", layout="wide")

st.title("SimFin Stock Explorer")

st.markdown("""
Welcome to the SimFin Stock Explorer web application.

Navigate using the sidebar to:
- View general company information
- Explore share price history
- Access financial statements
""")

st.write("Hot Stocks")

df = api.companies()

df.dropna(inplace=True)

df = df[df['ticker'].isin(['TSLA','NVDA','META', 'XOM', 'CRM', 'MSFT', 'ONON', 'AMZN', 'JPM'])][['name', 'ticker', 'industryName', 'sectorName']]

st.dataframe(df.head(15))

st.info("Developed for the Automated Trading System Project")
