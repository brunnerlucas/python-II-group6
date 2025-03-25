import streamlit as st
from PySimFin import PySimFin
import requests
import yfinance as yf

st.title("General Company Information")

api = PySimFin()

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", value="AAPL")

if st.button("Get Company Info"):
    df = api.get_general_data(ticker)
    df_similar = api.companies()
    if not df.empty:
        company_name = df['name'].iloc[0]
        st.subheader(f"Company: {company_name}")

        # --- Try to get domain using yfinance ---
        try:
            company_yf = yf.Ticker(ticker)
            domain = company_yf.info.get("website", None)
        except Exception as e:
            domain = None

        # --- If we get a domain, try to display the logo ---
        logo_displayed = False
        if domain and isinstance(domain, str):
            domain_base = domain.replace("https://", "").replace("http://", "").split("/")[0]
            logo_url = f"https://logo.clearbit.com/{domain_base}"

            try:
                response = requests.get(logo_url, timeout=3)
                if response.status_code == 200:
                    st.image(logo_url, width=100)
                    logo_displayed = True
            except:
                pass

        if not logo_displayed:
            st.info("🔍 Logo not found using company domain.")

        # --- Display general info ---
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
        st.subheader("Similar Companies:")
        similar_companies = df_similar[df_similar['sectorCode'] == df['sectorCode'].iloc[0]]['name'].sample(5).tolist()
        st.write(", ".join(similar_companies))
    else:
        st.warning("No company information found.")
