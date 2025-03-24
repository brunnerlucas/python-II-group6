import streamlit as st
from PySimFin import PySimFin

st.title("General Company Information")

api = PySimFin()

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", value="AAPL")

if st.button("Get Company Info"):
    df = api.get_general_data(ticker)
    if not df.empty:
        st.dataframe(df)
    else:
        st.warning("No company information found.")
