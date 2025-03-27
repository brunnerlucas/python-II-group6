## python-II-group6
# Stock Prediction and Recommendation App

## Contributors

- Lucas Brunner  
- Pablo Camacho  
- Isabella Dao  
- Yousef Joukhar  
- Diego López  

Project Overview

This Streamlit application allows users to analyze, compare, and predict stock prices using real-time financial data and machine learning. It integrates with the SimFin API to retrieve company metadata, daily stock prices, and financial statements, and uses an XGBoost model to generate next-day price predictions.

Key features include:

- Viewing general stock and industry information  
- Visualizing stock price trends and financials  
- Forecasting next-day stock prices using an XGBoost regression model  
- Simulating trading strategies and showing potential investment returns  
- Delivering buy/sell recommendations based on model predictions  

The project combines back-end data handling, machine learning, and a user-friendly Streamlit dashboard.

-Project Structure

.├── PySimFin.py # Main class for SimFin API access, modeling, and strategy logic
├── streamlit_app.py # Streamlit interface with multi-page layout
├── models/ # Pre-trained XGBoost models for selected stocks
├── data/ # Local storage for data snapshots (if needed)

---

 Technologies Used

- Python  
- Streamlit  
- SimFin API  
- XGBoost (regression model)  
- Plotly (visualizations)  
- pandas / numpy  

---

 ETL Pipeline

Most of the ETL (Extract, Transform, Load) process is handled via the SimFin API, which provides clean and structured data. Additional steps include:

 Data Selection & Filtering

- Only selected stock tickers are used  
- Training period is reduced to focus on relevant dates  

 Feature Engineering

- Log Returns  
- Rolling Windows (moving averages, lagged features)  
- Custom metrics for improved prediction accuracy  

---

 Machine Learning Model
 XGBoost Regressor

- Predicts next-day stock prices  
- Uses engineered features like log returns, rolling statistics, and lags  
- Trained per stock and saved locally to speed up demo performance  
- If no saved model exists, it trains one on the fly  

Trading Strategy

The strategy is based on predicted percentage change from the model:

- If the predicted increase exceeds a threshold → Buy  
- If the predicted decrease exceeds a threshold → Sell  
- A trade log is saved and shown to the user  
- Displays total return over a year with a $10,000 simulated investment  



Streamlit Dashboard

The user interface is divided into five pages:

 1. Opening Page  
- Overview of app functionality  
- (Previously included Hot Stocks from Yahoo Finance, later removed)

 2. General Information Page  
- Displays company details, industry, and similar stocks  
- Includes company logo for visual reference  

3. Share Prices Page  
- Plots selected stocks’ historical prices  
- Allows single or comparative view  

4. Financial Statements Page  
- Visualizes annual or quarterly income statements  
- Graphs revenue vs. net income (single or comparative)  

5. Prediction and Recommendation Page  
- Runs or loads XGBoost model for selected stock  
- Shows price forecast with historical context  
- Provides trading recommendations with detailed return metrics  

To launch the app:

```bash
streamlit run streamlit_app.py



**All the Data will be stored locally and not be held here on github**
    
# 🚀 Git Branching Strategy

## 🌳 Main Branches

### 🟢 `main` (Production-Ready Code)
- This branch should always have **stable, tested, and deployable code**.
- No one should push directly to `main`**.

### 🟡 `dev` (Integration & Testing)
- This is the **main working branch** where all feature branches are merged before going to `main`.
- Team members should **branch off `dev`** and create **feature branches** i.e. feature/rf_model .

## 🌱 Feature Branches (One per Task)
Each team member should create a **feature branch** off `dev`, named descriptively:


**API_KEY=SECRET** 
</code>
The API Key can be accessed on this page: https://www.simfin.com/en/

