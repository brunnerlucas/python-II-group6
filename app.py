import streamlit as st
from PySimFin import PySimFin

# Page setup
st.set_page_config(page_title="SimFin Stock Explorer", layout="wide")

# Custom background color using HTML
st.markdown(
    """
    <style>
    body {
        background-color: #f7f9fc;
    }
    .main-title {
        text-align: center;
        font-size: 2.5em;
        font-weight: 700;
        margin-bottom: 10px;
        border-bottom: 2px solid #ccc;
        padding-bottom: 0.5rem;
    }
    .section-title {
        font-size: 1.3em;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    .stButton>button {
        background-color: #ffffff;
        border: 1px solid #ccc;
        padding: 0.3rem 1rem;
        font-size: 1rem;
        border-radius: 6px;
    }
    .stButton>button:hover {
        background-color: #f0f0f0;
        border: 1px solid #999;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Page title
st.markdown('<div class="main-title">SimFin Stock Explorer</div>', unsafe_allow_html=True)

st.markdown("Welcome to the **SimFin Stock Explorer**, your dashboard for analyzing and predicting market behavior.")

# --- Navigation Section ---
st.markdown("### Explore the App")

# Section 1
st.markdown('<div class="section-title">General Company Info</div>', unsafe_allow_html=True)
st.markdown("Explore detailed company profiles including name, sector, industry, market, employee count, and description.")
if st.button("Go to General Info"):
    st.switch_page("pages/1_General_Info.py")

# Section 2
st.markdown('<div class="section-title">Share Prices</div>', unsafe_allow_html=True)
st.markdown("View and compare historical share prices of companies. Plot adjusted or last closing prices with export options.")
if st.button("Go to Share Prices"):
    st.switch_page("pages/2_Share_Prices.py")

# Section 3
st.markdown('<div class="section-title">Financial Statements</div>', unsafe_allow_html=True)
st.markdown("Access and compare income statements including revenue, expenses, net earnings, and more.")
if st.button("Go to Financial Statements"):
    st.switch_page("pages/3_Financial_Statements.py")

# Section 4
st.markdown('<div class="section-title">Stock Prediction</div>', unsafe_allow_html=True)
st.markdown("Use machine learning models to predict the next-day market behavior for selected tickers.")
if st.button("Go to Prediction"):
    st.switch_page("pages/4_Prediction.py")

# --- Hot Stocks Section ---
st.markdown("---")
st.markdown("### Hot Stocks")

api = PySimFin()
df = api.companies()
df.dropna(inplace=True)
df = df[df['ticker'].isin(['TSLA', 'NVDA', 'META', 'XOM', 'CRM', 'MSFT', 'ONON', 'AMZN', 'JPM'])]
df = df[['name', 'ticker', 'industryName', 'sectorName']]
st.dataframe(df.head(15), use_container_width=True)

# Footer
st.markdown("---")
st.info("🔧 Developed for the Automated Trading System Project")
