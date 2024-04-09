import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://t4.ftcdn.net/jpg/06/46/25/31/360_F_646253114_zBxkO87yubUkKjzyrXlaSBjodaC8kKHg.jpg");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# st.markdown("""
#     <h1 style='text-align: center; color: white;'>STOCK PRICE PREDICTION</h1>
#     """, unsafe_allow_html=True)

# Load your pre-trained model
# model = load_model('your_model.h5')  # Make sure to have a trained model saved and load it appropriately


# Streamlit interface
st.title('Stock Price Prediction')

# User inputs
stock_symbol = st.text_input('Enter Stock Symbol', 'AAPL')

# Date range picker
start_date = st.date_input('Start Date', dt.date(2020, 1, 1))
end_date = st.date_input('End Date', dt.date.today())

# Fetch stock data
@st.cache
def get_data(symbol, start, end):
    df = pdr.get_data_tiingo(symbol, start=start, end=end, api_key='5c8d4db97b48a803cff150a77ff468a2786f8483')
    df.to_csv(f'{symbol}.csv')
    return pd.read_csv(f'{symbol}.csv')

if st.button('Fetch Data'):
    df = get_data(stock_symbol, start_date, end_date)
    st.write(df.head())

    # Plot closing price
    plt.figure(figsize=(10,4))
    plt.plot(pd.to_datetime(df['date']), df['close'])
    plt.title(f'Closing Price of {stock_symbol}')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (USD)')
    st.pyplot(plt)

# Placeholder for future steps, e.g., displaying predictions

# Note: The LSTM model training and prediction process would typically be handled offline or separately
# due to the computational and time costs involved. You could trigger model predictions based on user inputs
# if the model is pre-trained and saved.
