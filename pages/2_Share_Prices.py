import streamlit as st
from PySimFin import PySimFin
import plotly.graph_objects as go
import pandas as pd
from io import BytesIO

st.title("Share Price History")

api = PySimFin()

ticker = st.text_input("Stock Ticker", value="AAPL")
start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")

# Compare with another company
compare = st.checkbox("Compare with another company")
ticker_2 = None
if compare:
    ticker_2 = st.text_input("Second Ticker (for comparison)", value="MSFT")

if st.button("Get Share Prices"):
    df_1 = api.get_share_prices(ticker, start=start_date.isoformat(), end=end_date.isoformat())

    df_2 = None
    if compare and ticker_2:
        df_2 = api.get_share_prices(ticker_2, start=start_date.isoformat(), end=end_date.isoformat())

    if df_1.empty:
        st.warning(f"No share price data found for {ticker.upper()}.")
    else:
        # --- Clean and prepare df_1 ---
        if "Dividend Paid" in df_1.columns:
            df_1 = df_1.drop(columns=["Dividend Paid"])
        st.subheader(f"Price Data for {ticker.upper()}")
        st.dataframe(df_1.head(10))

        # --- Clean and prepare df_2 ---
        if compare and ticker_2 and df_2 is not None and not df_2.empty:
            if "Dividend Paid" in df_2.columns:
                df_2 = df_2.drop(columns=["Dividend Paid"])
            st.subheader(f"Price Data for {ticker_2.upper()}")
            st.dataframe(df_2.head(10))

        # --- Excel export ---
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_1.to_excel(writer, sheet_name=f"{ticker.upper()}", index=False)
            if compare and ticker_2 and df_2 is not None and not df_2.empty:
                df_2.to_excel(writer, sheet_name=f"{ticker_2.upper()}", index=False)
        st.download_button(
            label="Download as Excel",
            data=output.getvalue(),
            file_name=f"{ticker.upper()}_share_prices.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # --- Plotting ---
        fig = go.Figure()

        # Helper to get the best price column
        def get_price_column(df):
            if "Adjusted Closing Price" in df.columns:
                return "Adjusted Closing Price"
            elif "Last Closing Price" in df.columns:
                return "Last Closing Price"
            return None

        price_col_1 = get_price_column(df_1)
        if price_col_1 and "Date" in df_1.columns:
            df_1["Date"] = pd.to_datetime(df_1["Date"])
            fig.add_trace(go.Scatter(
                x=df_1["Date"],
                y=df_1[price_col_1],
                mode="lines",
                name=f"{ticker.upper()} - {price_col_1}",
                line=dict(color="blue")
            ))

        if compare and ticker_2 and df_2 is not None and not df_2.empty:
            price_col_2 = get_price_column(df_2)
            if price_col_2 and "Date" in df_2.columns:
                df_2["Date"] = pd.to_datetime(df_2["Date"])
                fig.add_trace(go.Scatter(
                    x=df_2["Date"],
                    y=df_2[price_col_2],
                    mode="lines",
                    name=f"{ticker_2.upper()} - {price_col_2}",
                    line=dict(color="red")
                ))

        if fig.data:
            fig.update_layout(
                title="Adjusted/Closing Price Over Time",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                hovermode="x unified",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No valid price data to plot.")
