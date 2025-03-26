# This Streamlit app displays general company information using SimFin and Yahoo Finance data.
# Users can input a stock ticker to retrieve details such as description, sector, market, logo, and similar companies.

import streamlit as st
import requests
import yfinance as yf
from PySimFin import PySimFin


# === Page Title ===
st.title("General Company Information")

# === Initialize SimFin API ===
api = PySimFin()

# === User Input ===
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", value="AAPL")


# === Helper Functions ===

# Prompt: Attempt to retrieve the company's logo using its website domain and the Clearbit API
def fetch_company_logo(ticker_symbol):
    try:
        company_yf = yf.Ticker(ticker_symbol)
        domain = company_yf.info.get("website", "")
        if domain:
            domain_base = domain.replace("https://", "").replace("http://", "").split("/")[0]
            logo_url = f"https://logo.clearbit.com/{domain_base}"
            response = requests.get(logo_url, timeout=3)
            if response.status_code == 200:
                return logo_url
    except:
        pass
    return None

# Prompt: Display key information about the selected company, including logo, description, industry, market, sector, and number of employees
def display_company_info(df):
    company_name = df['name'].iloc[0]
    st.subheader(f"Company: {company_name}")

    # --- Try displaying logo ---
    logo_url = fetch_company_logo(ticker)
    if logo_url:
        st.image(logo_url, width=100)
    else:
        st.info("🔍 Logo not found using company domain.")

    # --- Display company details ---
    st.subheader("Company Description:")
    st.write(df['companyDescription'].iloc[0])

    st.subheader("Industry:")
    st.write(df['industryName'].iloc[0])

    st.subheader("Main Market:")
    st.write(df['market'].iloc[0])

    st.subheader("Sector:")
    st.write(df['sectorName'].iloc[0])

    st.subheader("Employees:")
    st.write(f"{df['numEmployees'].iloc[0]:,}")


def display_similar_companies(df, df_all):
    st.subheader("Similar Companies:")
    sector_code = df['sectorCode'].iloc[0]
    similar = df_all[df_all['sectorCode'] == sector_code]['name']
    if not similar.empty:
        st.write(", ".join(similar.sample(min(5, len(similar))).tolist()))
    else:
        st.write("No similar companies found.")


# === Main Logic ===
if st.button("Get Company Info"):
    df = api.get_general_data(ticker)
    df_similar = api.companies()

    if not df.empty:
        display_company_info(df)
        display_similar_companies(df, df_similar)
    else:
        st.warning("No company information found.")

# This code has been refactored with the assistance of ChatGPT to enhance structure,
# modularity, and adherence to clean coding principles.
