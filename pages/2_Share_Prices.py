import streamlit as st
from PySimFin import PySimFin

st.title("Share Price History")

api = PySimFin()

ticker = st.text_input("Stock Ticker", value="AAPL")
start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")

if st.button("Get Share Prices"):
    df = api.get_share_prices(ticker, start=start_date.isoformat(), end=end_date.isoformat())
    if not df.empty:
        st.dataframe(df.head(10))
        if "close" in df.columns:
            st.line_chart(df.set_index("date")["close"])
    else:
        st.warning("No share price data found.")
