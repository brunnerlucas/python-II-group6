# This Streamlit app lets users visualize and compare historical share prices for one or two companies.
# It retrieves data from the SimFin API, displays it in a table, provides Excel export, and generates interactive plots using Plotly.

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from io import BytesIO
from PySimFin import PySimFin


# === Page Title ===
st.title("📈 Share Price History")

# === Initialize API ===
api = PySimFin()

# === Inputs ===
ticker = st.text_input("Stock Ticker", value="AAPL")
start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")

compare = st.checkbox("Compare with another company")
ticker_2 = st.text_input("Second Ticker (for comparison)", value="MSFT") if compare else None


# === Helper Functions ===

# Prompt: Identify the most relevant column for price visualization (adjusted or last closing)
def get_price_column(df):
    """Selects the best price column available."""
    if "Adjusted Closing Price" in df.columns:
        return "Adjusted Closing Price"
    elif "Last Closing Price" in df.columns:
        return "Last Closing Price"
    return None

# Prompt: Drop unnecessary columns to clean up the display
def clean_dataframe(df):
    """Removes unnecessary columns."""
    return df.drop(columns=["Dividend Paid"], errors="ignore")

# Prompt: Allow users to export one or two company (depending if they are comparing or not) price histories as an Excel file
def export_to_excel(df_1, df_2=None, ticker_1=None, ticker_2=None):
    """Exports price data to Excel."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_1.to_excel(writer, sheet_name=ticker_1.upper(), index=False)
        if df_2 is not None:
            df_2.to_excel(writer, sheet_name=ticker_2.upper(), index=False)
    return output.getvalue()

# Prompt: Generate an interactive line plot using Plotly to visualize stock prices over time
def plot_prices(df_1, ticker_1, df_2=None, ticker_2=None):
    """Plots share prices for one or two companies."""
    fig = go.Figure()

    price_col_1 = get_price_column(df_1)
    if price_col_1:
        df_1["Date"] = pd.to_datetime(df_1["Date"])
        fig.add_trace(go.Scatter(
            x=df_1["Date"],
            y=df_1[price_col_1],
            mode="lines",
            name=f"{ticker_1.upper()} - {price_col_1}",
            line=dict(color="blue")
        ))

    if df_2 is not None:
        price_col_2 = get_price_column(df_2)
        if price_col_2:
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


# === Main Logic ===
if st.button("Get Share Prices"):
    df_1 = api.get_share_prices(ticker, start=start_date.isoformat(), end=end_date.isoformat())
    df_2 = api.get_share_prices(ticker_2, start=start_date.isoformat(), end=end_date.isoformat()) if compare and ticker_2 else None

    if df_1.empty:
        st.warning(f"No share price data found for {ticker.upper()}.")
    else:
        df_1 = clean_dataframe(df_1)
        st.subheader(f"📊 Price Data for {ticker.upper()}")
        st.dataframe(df_1.head(10))

        if compare and df_2 is not None and not df_2.empty:
            df_2 = clean_dataframe(df_2)
            st.subheader(f"📊 Price Data for {ticker_2.upper()}")
            st.dataframe(df_2.head(10))

        # === Excel Export ===
        excel_data = export_to_excel(df_1, df_2, ticker, ticker_2)
        st.download_button(
            label="📥 Download as Excel",
            data=excel_data,
            file_name=f"{ticker.upper()}_share_prices.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # === Plotting ===
        plot_prices(df_1, ticker, df_2, ticker_2 if df_2 is not None else None)

# This code has been refactored with the assistance of ChatGPT to enhance structure,
# modularity, and adherence to clean coding principles.

