import streamlit as st
from PySimFin import PySimFin

st.title("Financial Statements")

api = PySimFin()

ticker = st.text_input("Stock Ticker", value="AAPL")
statement_type = st.selectbox("Statement Type", ["Derived", "Income", "Balance", "Cashflow"])
period = st.selectbox("Period", ["FY", "Q1", "Q2", "Q3", "Q4"])
fyear = st.text_input("Fiscal Year (e.g., 2022)", value="2022")

if st.button("Get Financial Data"):
    df = api.get_financial_statement(ticker, statements=statement_type, period=period, fyear=fyear)
    if not df.empty:
        st.dataframe(df.head(10))
    else:
        st.warning("No financial statement data found.")
