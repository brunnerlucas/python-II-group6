import streamlit as st
import pandas as pd
from datetime import datetime
from PySimFin import PySimFin
import plotly.graph_objects as go

api = PySimFin()


# Input section
ticker = st.text_input("Enter stock ticker (e.g., AAPL):", "AAPL")

# Start and End Date Pickers
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start date:", datetime(2024, 1, 1))
with col2:
    end_date = st.date_input("End date (last day for training):", datetime.today())



if start_date >= end_date:
    st.error("Start date must be earlier than end date.")
elif st.button("Predict Next Close Price"):
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    #getting the data for the prediction
    df = api.get_share_prices(ticker, start=start_str, end=end_str)

    if df.empty:
        st.warning("No data found for this ticker.")
        st.stop()


    # Normalize and prepare columns
    df.columns = [col.strip().title() for col in df.columns]
    df['Date'] = pd.to_datetime(df['Date'])
    if 'Adjusted Closing Price' in df.columns:
        df = df.rename(columns={'Adjusted Closing Price': 'Close'})
    if 'Ticker' not in df.columns:
        df['Ticker'] = ticker


    last_date = df['Date'].max()

    try:
        prediction_df = api.train_xgboost_model(df, last_date)
    except ValueError as e:
        st.warning(str(e))
        st.stop()
    next_day = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=1, freq='B')[0]


    next_day = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=1, freq='B')[0]
    next_day_str = next_day.strftime('%Y-%m-%d')

    # Fetch actual close for the prediction day
    actual_row = api.get_share_prices(ticker, start=next_day_str, end=next_day_str)
    actual_price = None

    if not actual_row.empty:
        actual_row.columns = [col.strip().title() for col in actual_row.columns]

        if 'Adjusted Closing Price' in actual_row.columns:
            actual_price = actual_row['Adjusted Closing Price'].values[0]


    # Add to prediction DataFrame
    prediction_df['Actual_Close'] = actual_price

    # Format the 'Date' column to show only the date (no time)
    prediction_df['Date'] = prediction_df['Date'].dt.date

    # Show results
    st.subheader(f"Prediction for {ticker} on {next_day.date()}:")
    st.dataframe(prediction_df)



    # Plot chart
    st.subheader("Recent Closing Prices")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Historical Close"))
    fig.add_trace(go.Scatter(x=prediction_df['Date'], y=prediction_df['Predicted_Close'], name="Predicted Close", mode='markers+lines'))
    if actual_price is not None:
        fig.add_trace(go.Scatter(x=[next_day], y=[actual_price], name="Actual Close", mode='markers'))
    fig.update_layout(xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)


    strategy_df = api.simulate_trading_strategy(df, api.train_xgboost_model, ticker)

    st.subheader("📈 Strategy Equity Curve")
    st.line_chart(strategy_df.set_index("Date")["Capital"])

    total_return = strategy_df['Capital'].iloc[-1] - 10000
    st.metric("💰 Total Profit", f"${total_return:.2f}")
