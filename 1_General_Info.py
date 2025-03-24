import streamlit as st
from PySimFin import PySimFin
import requests

st.title("General Company Information")

api = PySimFin()

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", value="AAPL")

if st.button("Get Company Info"):
    df = api.get_general_data(ticker)
    df2 = api.companies()
    if not df.empty:
        # Fetch logo using Clearbit Logo API based on company name
        company_name = df['name'].iloc[0]
        company_name_cleaned = company_name.replace('INC', '').replace('CORP', '').replace(' ', '').lower()
        logo_url = f"https://logo.clearbit.com/{company_name_cleaned}.com"
        st.image(logo_url, width=100)
        st.subheader("Name:")
        st.write(df['name'].iloc[0])
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
        similar_companies = df2[df2['sectorCode'] == df['sectorCode'].iloc[0]]['name'].sample(5).tolist()
        st.write(", ".join(similar_companies))
    else:
        st.warning("No company information found.")