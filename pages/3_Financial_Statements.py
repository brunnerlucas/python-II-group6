# This Streamlit page displays a summarized Income Statement by combining data from SimFin's 
# Profit & Loss and Cash Flow reports. Users can view financials by year or quarter, compare two companies, 
# visualize trends, and export the results as Excel files.

import streamlit as st
from PySimFin import PySimFin
import pandas as pd
import plotly.graph_objects as go
from io import BytesIO

# === Page Title ===
st.title("📊 Financial Statements - Income Statement")

api = PySimFin()

# === Inputs ===
ticker = st.text_input("Stock Ticker", value="AAPL")  # Prompt: User input for primary stock ticker
compare = st.checkbox("Compare with another company")  # Prompt: Enable option to compare with a second company
ticker_2 = st.text_input("Second Ticker (for comparison)", value="MSFT") if compare else None

period = st.selectbox("Period", ["FY", "Q1", "Q2", "Q3", "Q4"])  # Prompt: Select full-year or quarterly data
years_range = [""] + [str(year) for year in range(2018, 2025)]  # Prompt: Build a dynamic year range for filtering
start_year = st.selectbox("Start Year (optional)", years_range)
end_year = st.selectbox("End Year (optional)", years_range)

# === Helpers ===

# Prompt: Apply visual styling to emphasize key totals like Gross Profit and Net Earnings
def highlight_totals(row):
    keywords = ["gross profit", "total expenses", "earnings before tax", "net earnings"]
    return ['font-weight: bold; background-color: #f0f0f0' if any(k in row.name.lower() for k in keywords) else '' for _ in row]
    
# Prompt: Format the income statement nicely using pandas Styler
def style_income_df(df):
    return df.fillna(0).style \
        .format("{:,.0f}") \
        .apply(highlight_totals, axis=1) \
        .set_properties(**{'text-align': 'right', 'font-family': 'Calibri', 'font-size': '14px'}) \
        .set_table_styles([{'selector': 'thead', 'props': [('font-weight', 'bold'), ('background-color', '#dbe5f1')]}])

# Prompt: Construct a simplified income statement using both profit & loss and cash flow data (was further adjusted)
def build_income_statement(df_pl, df_cf):
    df = pd.DataFrame()
    df["Revenue"] = df_pl.get("Revenue", pd.Series(0, index=df_pl.index))
    df["Cost of Goods Sold (COGS)"] = df_pl.get("Cost of revenue", pd.Series(0, index=df_pl.index)).abs()
    df["Gross Profit"] = df["Revenue"] - df["Cost of Goods Sold (COGS)"]
    df["Selling, General & Administrative (SG&A)"] = df_pl.get("Selling, General & Administrative", pd.Series(0, index=df_pl.index)).abs()
    df["Depreciation & Amortization"] = df_cf.get("Depreciation & Amortization", pd.Series(0, index=df_pl.index)).abs()

    if "Interest Expense" in df_pl.columns and df_pl["Interest Expense"].notna().any():
        df["Interest"] = df_pl["Interest Expense"].abs()
    elif "Operating Income (Loss)" in df_pl.columns and "Pretax Income (Loss)" in df_pl.columns:
        df["Interest"] = (df_pl["Operating Income (Loss)"] - df_pl["Pretax Income (Loss)"]).abs()
    else:
        df["Interest"] = pd.Series(0, index=df_pl.index)

    df["Total Expenses"] = df[["Selling, General & Administrative (SG&A)", "Depreciation & Amortization", "Interest"]].sum(axis=1)
    df["Earnings Before Tax"] = df["Gross Profit"] - df["Total Expenses"]
    df["Taxes"] = df_pl.get("Income Tax (Expense) Benefit, net", pd.Series(0, index=df_pl.index)).abs()
    df["Net Earnings"] = df["Earnings Before Tax"] - df["Taxes"]

    df = df.T
    df.columns = df_pl["Fiscal Year"].astype(str)
    df.index.name = "Line Item"
    return df

# Prompt: Generate a financial trend plot comparing either Revenue & Net Earnings for one company, or Net Earnings across two companies
def plot_trends(df_1, ticker_1, df_2=None, ticker_2=None):
    fig = go.Figure()

    if df_2 is not None:
        for df, name, color in zip([df_1, df_2], [ticker_1, ticker_2], ['red', 'blue']):
            chart_data = df.T[["Net Earnings"]].apply(pd.to_numeric, errors="coerce")
            fig.add_trace(go.Scatter(
                x=chart_data.index,
                y=chart_data.iloc[:, 0],
                name=f"{name.upper()} - Net Earnings",
                line=dict(color=color)
            ))
        fig.update_layout(title="Net Earnings Comparison", xaxis_title="Fiscal Year", yaxis_title="Amount (USD)")
    else:
        chart_data = df_1.T[["Revenue", "Net Earnings"]].apply(pd.to_numeric, errors="coerce")
        fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["Revenue"], name="Revenue", line=dict(color='red')))
        fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["Net Earnings"], name="Net Earnings", line=dict(color='blue')))
        fig.update_layout(title=f"{ticker_1.upper()} Revenue vs Net Earnings", xaxis_title="Fiscal Year", yaxis_title="Amount (USD)")

    st.plotly_chart(fig, use_container_width=True)

# Prompt: Export the income statements to an Excel file for download (one or two companies)
def download_excel(df_1, ticker_1, df_2=None, ticker_2=None):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_1.T.to_excel(writer, sheet_name=ticker_1.upper())
        if df_2 is not None:
            df_2.T.to_excel(writer, sheet_name=ticker_2.upper())
    st.download_button(
        label="📥 Download as Excel",
        data=output.getvalue(),
        file_name=f"{ticker_1.upper()}_income_statement.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# === Main Logic ===
if st.button("Get Income Statement"):
    start = start_year if start_year else None
    end = end_year if end_year else None

    df_pl = api.get_financial_statement(ticker, statements="pl", period=period, start=start, end=end)
    df_cf = api.get_financial_statement(ticker, statements="cf", period=period, start=start, end=end)

    if df_pl.empty:
        st.warning(f"No financial data found for {ticker.upper()}.")
    else:
        income_df = build_income_statement(df_pl, df_cf)
        st.subheader(f"📄 Income Statement – {ticker.upper()}")
        st.dataframe(style_income_df(income_df), use_container_width=True)

        income_df_2 = None
        if compare and ticker_2:
            df_pl_2 = api.get_financial_statement(ticker_2, "pl", period, start, end)
            df_cf_2 = api.get_financial_statement(ticker_2, "cf", period, start, end)
            if not df_pl_2.empty:
                income_df_2 = build_income_statement(df_pl_2, df_cf_2)
                st.subheader(f"📄 Income Statement – {ticker_2.upper()}")
                st.dataframe(style_income_df(income_df_2), use_container_width=True)
            else:
                st.warning(f"No data found for comparison ticker '{ticker_2.upper()}'.")

        download_excel(income_df, ticker, income_df_2, ticker_2 if income_df_2 is not None else None)

        with st.expander("📉 Show Financial Trends"):
            plot_trends(income_df, ticker, income_df_2, ticker_2 if income_df_2 is not None else None)

# This code has been refactored with the assistance of ChatGPT to enhance structure,
# modularity, and adherence to clean coding principles.
