import streamlit as st
from PySimFin import PySimFin
import pandas as pd
import plotly.graph_objects as go
from io import BytesIO

st.title("Financial Statements - Income Statement")

api = PySimFin()

ticker = st.text_input("Stock Ticker", value="AAPL")
compare = st.checkbox("Compare with another company")

ticker_2 = None
if compare:
    ticker_2 = st.text_input("Second Ticker (for comparison)", value="MSFT")

period = st.selectbox("Period", ["FY", "Q1", "Q2", "Q3", "Q4"])

# Optional start and end year
years_range = [""] + [str(year) for year in range(2018, 2025)]
start_year = st.selectbox("Start Year (optional)", options=years_range, index=0)
end_year = st.selectbox("End Year (optional)", options=years_range, index=0)

if st.button("Get Income Statement"):
    start = start_year if start_year else None
    end = end_year if end_year else None

    # Fetch data from both statements
    df_pl = api.get_financial_statement(ticker, statements="pl", period=period, start=start, end=end)
    df_cf = api.get_financial_statement(ticker, statements="cf", period=period, start=start, end=end)

    if df_pl.empty:
        st.warning("No financial data found.")
    else:
        income_df = pd.DataFrame()
        income_df["Revenue"] = df_pl.get("Revenue", pd.Series(0, index=df_pl.index))
        income_df["Cost of Goods Sold (COGS)"] = df_pl.get("Cost of revenue", pd.Series(0, index=df_pl.index)).abs()
        income_df["Gross Profit"] = income_df["Revenue"] - income_df["Cost of Goods Sold (COGS)"]

        # SG&A
        income_df["Selling, General & Administrative (SG&A)"] = df_pl.get("Selling, General & Administrative", pd.Series(0, index=df_pl.index)).abs()

        # Depreciation (from CF statement)
        if "Depreciation & Amortization" in df_cf.columns:
            income_df["Depreciation & Amortization"] = df_cf["Depreciation & Amortization"].abs()
        else:
            income_df["Depreciation & Amortization"] = pd.Series(0, index=df_pl.index)

        # Interest: direct or estimated fallback
        if "Interest Expense" in df_pl.columns and df_pl["Interest Expense"].notna().any():
            income_df["Interest"] = df_pl["Interest Expense"].abs()
        elif "Operating Income (Loss)" in df_pl.columns and "Pretax Income (Loss)" in df_pl.columns:
            income_df["Interest"] = (df_pl["Operating Income (Loss)"] - df_pl["Pretax Income (Loss)"]).abs()
        else:
            income_df["Interest"] = pd.Series(0, index=df_pl.index)

        # Total Expenses
        expense_cols = [
            "Selling, General & Administrative (SG&A)",
            "Depreciation & Amortization",
            "Interest"
        ]
        income_df["Total Expenses"] = income_df[expense_cols].sum(axis=1)

        # Earnings
        income_df["Earnings Before Tax"] = income_df["Gross Profit"] - income_df["Total Expenses"]
        income_df["Taxes"] = df_pl.get("Income Tax (Expense) Benefit, net", pd.Series(0, index=df_pl.index)).abs()
        income_df["Net Earnings"] = income_df["Earnings Before Tax"] - income_df["Taxes"]

        # Format for display
        income_df = income_df.T
        income_df.columns = df_pl["Fiscal Year"].astype(str)
        income_df.index.name = "Line Item"

        csv = income_df.to_csv().encode('utf-8')

        # Excel-like style
        def highlight_totals(row):
            if any(keyword in row.name.lower() for keyword in ["gross profit", "total expenses", "earnings before tax", "net earnings"]):
                return ['font-weight: bold; background-color: #f0f0f0'] * len(row)
            return [''] * len(row)

        styled_df = income_df.fillna(0).style\
            .format("{:,.0f}")\
            .apply(highlight_totals, axis=1)\
            .set_properties(**{
                'text-align': 'right',
                'font-family': 'Calibri',
                'font-size': '14px'
            })\
            .set_table_styles([{
                'selector': 'thead',
                'props': [('font-weight', 'bold'), ('background-color', '#dbe5f1')]
            }])

        #st.subheader("Formatted Income Statement")
        st.subheader(f"Income Statement – {ticker.upper()}")
        st.dataframe(styled_df, use_container_width=True)

        # --- Comparison logic ---
        if compare and ticker_2:
            df_pl_2 = api.get_financial_statement(ticker_2, statements="pl", period=period, start=start, end=end)
            df_cf_2 = api.get_financial_statement(ticker_2, statements="cf", period=period, start=start, end=end)
        
            if not df_pl_2.empty:
                income_df_2 = pd.DataFrame()
                income_df_2["Revenue"] = df_pl_2.get("Revenue", pd.Series(0, index=df_pl_2.index))
                income_df_2["Cost of Goods Sold (COGS)"] = df_pl_2.get("Cost of revenue", pd.Series(0, index=df_pl_2.index)).abs()
                income_df_2["Gross Profit"] = income_df_2["Revenue"] - income_df_2["Cost of Goods Sold (COGS)"]
                income_df_2["Selling, General & Administrative (SG&A)"] = df_pl_2.get("Selling, General & Administrative", pd.Series(0, index=df_pl_2.index)).abs()
                if "Depreciation & Amortization" in df_cf_2.columns:
                    income_df_2["Depreciation & Amortization"] = df_cf_2["Depreciation & Amortization"].abs()
                else:
                    income_df_2["Depreciation & Amortization"] = pd.Series(0, index=df_pl_2.index)
                if "Interest Expense" in df_pl_2.columns and df_pl_2["Interest Expense"].notna().any():
                    income_df_2["Interest"] = df_pl_2["Interest Expense"].abs()
                elif "Operating Income (Loss)" in df_pl_2.columns and "Pretax Income (Loss)" in df_pl_2.columns:
                    income_df_2["Interest"] = (df_pl_2["Operating Income (Loss)"] - df_pl_2["Pretax Income (Loss)"]).abs()
                else:
                    income_df_2["Interest"] = pd.Series(0, index=df_pl_2.index)
                expense_cols_2 = ["Selling, General & Administrative (SG&A)", "Depreciation & Amortization", "Interest"]
                income_df_2["Total Expenses"] = income_df_2[expense_cols_2].sum(axis=1)
                income_df_2["Earnings Before Tax"] = income_df_2["Gross Profit"] - income_df_2["Total Expenses"]
                income_df_2["Taxes"] = df_pl_2.get("Income Tax (Expense) Benefit, net", pd.Series(0, index=df_pl_2.index)).abs()
                income_df_2["Net Earnings"] = income_df_2["Earnings Before Tax"] - income_df_2["Taxes"]
                income_df_2 = income_df_2.T
                income_df_2.columns = df_pl_2["Fiscal Year"].astype(str)
                income_df_2.index.name = "Line Item"
        
                st.subheader(f"Income Statement – {ticker_2.upper()}")
                styled_df_2 = income_df_2.fillna(0).style\
                    .format("{:,.0f}")\
                    .apply(highlight_totals, axis=1)\
                    .set_properties(**{
                        'text-align': 'right',
                        'font-family': 'Calibri',
                        'font-size': '14px'
                    })\
                    .set_table_styles([{
                        'selector': 'thead',
                        'props': [('font-weight', 'bold'), ('background-color', '#dbe5f1')]
                    }])
                
                st.dataframe(styled_df_2, use_container_width=True)

            else:
                st.warning(f"No data found for comparison ticker '{ticker_2.upper()}'.")

        # Excel export (replace your current download_button logic with this)
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            income_df.T.to_excel(writer, sheet_name=f"{ticker.upper()}")
        
            if compare and ticker_2 and 'income_df_2' in locals():
                income_df_2.T.to_excel(writer, sheet_name=f"{ticker_2.upper()}")
        
        # No need for writer.save()
        processed_data = output.getvalue()
        
        st.download_button(
            label="Download as Excel",
            data=processed_data,
            file_name=f"{ticker.upper()}_income_statement.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
                
        with st.expander("Show Financial Trends"):
            if compare and ticker_2 and 'income_df_2' in locals():
                # Only plot Net Earnings for both companies
                chart_data_1 = income_df.T[["Net Earnings"]].copy()
                chart_data_1.columns = [f"{ticker.upper()} - Net Earnings"]
                chart_data_1 = chart_data_1.apply(pd.to_numeric, errors="coerce")
        
                chart_data_2 = income_df_2.T[["Net Earnings"]].copy()
                chart_data_2.columns = [f"{ticker_2.upper()} - Net Earnings"]
                chart_data_2 = chart_data_2.apply(pd.to_numeric, errors="coerce")
        
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=chart_data_1.index,
                    y=chart_data_1.iloc[:, 0],
                    name=chart_data_1.columns[0],
                    line=dict(color='red')
                ))
                fig.add_trace(go.Scatter(
                    x=chart_data_2.index,
                    y=chart_data_2.iloc[:, 0],
                    name=chart_data_2.columns[0],
                    line=dict(color='blue')
                ))
                fig.update_layout(title="Net Earnings Comparison", xaxis_title="Fiscal Year", yaxis_title="Amount (USD)")
                st.plotly_chart(fig, use_container_width=True)
        
            else:
                chart_data = income_df.T[["Revenue", "Net Earnings"]].copy()
                chart_data = chart_data.apply(pd.to_numeric, errors="coerce")
        
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=chart_data.index,
                    y=chart_data["Revenue"],
                    name="Revenue",
                    line=dict(color='red')
                ))
                fig.add_trace(go.Scatter(
                    x=chart_data.index,
                    y=chart_data["Net Earnings"],
                    name="Net Earnings",
                    line=dict(color='blue')
                ))
                fig.update_layout(title=f"{ticker.upper()} Revenue vs Net Earnings", xaxis_title="Fiscal Year", yaxis_title="Amount (USD)")
                st.plotly_chart(fig, use_container_width=True)



