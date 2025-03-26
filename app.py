# This script defines the main Streamlit dashboard for the SimFin Stock Explorer.
# It links to different app sections and displays live top gaining stocks from Yahoo Finance.

import streamlit as st
from PySimFin import PySimFin
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ------------------------ Page Setup ------------------------
st.set_page_config(page_title="SimFin Stock Explorer", layout="wide")

# Prompt: Apply custom styling to make the app look cleaner and more user-friendly
def apply_custom_css(): 
    st.markdown("""
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
    """, unsafe_allow_html=True)

apply_custom_css()

# ------------------------ Header ------------------------
st.markdown('<div class="main-title">SimFin Stock Explorer</div>', unsafe_allow_html=True)
st.markdown("Welcome to the **SimFin Stock Explorer**, your dashboard for analyzing and predicting market behavior.")

# ------------------------ Navigation ------------------------
st.markdown("### Explore the App")

# Prompt: Display different sections of the app and link them to separate pages
def render_section(title, description, page_path): 
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(description)
    if st.button(f"Go to {title}"):
        st.switch_page(page_path)

render_section("General Company Info", "Explore detailed company profiles including name, sector, industry, market, employee count, and description.", "pages/1_General_Info.py")
render_section("Share Prices", "View and compare historical share prices of companies. Plot adjusted or last closing prices with export options.", "pages/2_Share_Prices.py")
render_section("Financial Statements", "Access and compare income statements including revenue, expenses, net earnings, and more.", "pages/3_Financial_Statements.py")
render_section("Stock Prediction", "Use machine learning models to predict the next-day market behavior for selected tickers.", "pages/4_Prediction.py")

# ------------------------ Hot Stocks Section ------------------------
st.markdown("---")
st.markdown("### 🔥 Top 10 Hottest Stocks Today")

#Prompt: # Scrape Yahoo Finance to display the top 10 hottest stocks (biggest gainers today)
def fetch_hot_stocks(): 
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        url = "https://finance.yahoo.com/gainers"
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table")
        rows = table.find("tbody").find_all("tr")

        data = []
        for row in rows[:10]:
            cols = row.find_all("td")
            data.append({
                "Name": cols[1].text.strip(),
                "Ticker": cols[0].text.strip(),
                "Change %": cols[4].text.strip()
            })

        df = pd.DataFrame(data)
        df['Change %'] = df['Change %'].str.replace('%', '', regex=False).str.replace('+', '', regex=False).astype(float)
        df = df.sort_values(by='Change %', ascending=False)
        df['Change %'] = df['Change %'].map(lambda x: f"+{x:.2f}%")
        return df

    except Exception as e:
        st.error("❌ Could not fetch Yahoo Finance data.")
        st.exception(e)
        return None

hot_stocks_df = fetch_hot_stocks()
if hot_stocks_df is not None:
    st.dataframe(hot_stocks_df, use_container_width=True)

# ------------------------ Footer ------------------------
st.markdown("---")
st.info("Developed by Hottest Group LLM")

# This code has been refactored with the assistance of ChatGPT to enhance structure,
# modularity, and adherence to clean coding principles.
