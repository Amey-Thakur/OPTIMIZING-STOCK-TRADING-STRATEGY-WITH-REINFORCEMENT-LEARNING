"""
Project: Optimizing Stock Trading Strategy With Reinforcement Learning
Authors: Amey Thakur & Mega Satish
Reference: https://github.com/Amey-Thakur/OPTIMIZING-STOCK-TRADING-STRATEGY-WITH-REINFORCEMENT-LEARNING
License: MIT

Description:
This script contains the Main Application logic served via Streamlit.
It loads the pre-trained Q-Learning model (model.pkl), processes user-selected
stock data, simulates the trading strategy on unseen data, and visualizes
the portfolio performance using interactive Plotly charts.
"""

import numpy as np
import pandas as pd
from pandas._libs.missing import NA
import streamlit as st
import time
import plotly.graph_objects as go
import pickle as pkl

# ==========================================
# 1. Data Processing Logic
# ==========================================
# @st.cache(persist=True)
def data_prep(data, name):    
    """
    Prepares the dataset for the selected stock ticker.
    
    Args:
        data (pd.DataFrame): The raw dataset.
        name (str): The specific stock name selected by the user.
        
    Returns:
        pd.DataFrame: A clean dataframe with computed Moving Averages (5-day & 1-day).
    """
    df = pd.DataFrame(data[data['Name'] == name])
    df.dropna(inplace=True)    
    df.reset_index(drop=True, inplace=True)
    
    # Calculate Moving Averages (Technical Indicators)
    # These indicators form the basis of the State Space for the RL agent.
    df['5day_MA'] = df['close'].rolling(5).mean()
    df['1day_MA'] = df['close'].rolling(1).mean()
    
    # Handle initial NaN values
    df.loc[:4, '5day_MA'] = 0
    
    return df

# ==========================================
# 2. Agent Logic (Inference)
# ==========================================
# @st.cache(persist=True)
def get_state(long_ma, short_ma, t):
    """
    Determines the current state of the market based on MA crossovers.
    
    Returns a tuple (Trend, Position) matching the Q-Table structure used during training.
    """
    if short_ma < long_ma:
        if t == 1:
            return (0, 1) # Bearish, Cash
        else:
            return (0, 0) # Bearish, Stock
    
    elif short_ma > long_ma:
        if t == 1:
            return (1, 1) # Bullish, Cash
        else:
            return (1, 0) # Bullish, Stock
            
    return (0, 1) # Default

# @st.cache(persist=True)
def trade_t(num_of_stocks, port_value, current_price):
    """
    Checks if a trade (Buy) is financially feasible.
    """
    if num_of_stocks >= 0:
        if port_value > current_price:
            return 1 # Can Buy
        else: return 0
    else:
        if port_value > current_price:
            return 1
        else: return 0

# @st.cache(persist=True)
def next_act(state, qtable, epsilon, action=3):
    """
    Decides the next action based on the trained Q-Table.
    
    During inference (testing), epsilon is typically 0 (pure exploitation),
    meaning the agent always chooses the optimal action learned during training.
    """
    if np.random.rand() < epsilon:
        action = np.random.randint(action)
    else:
        action = np.argmax(qtable[state])
    return action


# @st.cache(persist=True)
def test_stock(stocks_test, q_table, invest):
    """
    Runs a simulation of the trading strategy on the selected stock.
    
    Args:
        stocks_test (pd.DataFrame): The stock data to test on.
        q_table (np.array): The loaded reinforcement learning model.
        invest (int): Initial investment amount.
        
    Returns:
        list: A time-series list of net worth values over the simulation period.
    """
    num_stocks = 0
    epsilon = 0 # No exploration during testing/inference
    net_worth = [invest]
    np.random.seed()

    for dt in range(len(stocks_test)):
        long_ma = stocks_test.iloc[dt]['5day_MA']
        short_ma = stocks_test.iloc[dt]['1day_MA']
        close_price = stocks_test.iloc[dt]['close']
        
        # Determine Current State
        t = trade_t(num_stocks, net_worth[-1], close_price)
        state = get_state(long_ma, short_ma, t)
        
        # Agent chooses action
        action = next_act(state, q_table, epsilon)

        if action == 0: # Buy
            num_stocks += 1
            to_append = net_worth[-1] - close_price
            net_worth.append(np.round(to_append, 1))
            
        elif action == 1: # Sell
            num_stocks -= 1
            to_append = net_worth[-1] + close_price
            net_worth.append(np.round(to_append, 1))
        
        elif action == 2: # Hold
            to_append = net_worth[-1] + close_price # Mark-to-market valuation
            net_worth.append(np.round(to_append, 1))
            
        # Check for next state existence
        try:
            next_state = get_state(stocks_test.iloc[dt+1]['5day_MA'], stocks_test.iloc[dt+1]['1day_MA'], t)
        except:
            break

    return net_worth


# ==========================================
# 3. Streamlit Interface
# ==========================================
def fun():
    # Reading the Dataset
    # Ensure all_stocks_5yr.csv is in the working directory
    data = pd.read_csv('all_stocks_5yr.csv')
    names = list(data['Name'].unique())
    names.insert(0, "<Select Names>")

    st.title("Optimizing Stock Trading Strategy With Reinforcement Learning")

    st.sidebar.title("Choose Stock and Investment")
    st.sidebar.subheader("Choose Company Stocks")
    
    # User Input: Select Stock
    stock = st.sidebar.selectbox("(*select one stock only)", names, index=0)
    
    if stock != "<Select Names>":
        stock_df = data_prep(data, stock)
        
        # Sidebar Checkbox: Plot Data Trend
        if st.sidebar.button("Show Stock Trend", key=1):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=stock_df['date'], 
                y=stock_df['close'],
                mode='lines',
                name='Stock_Trend',
                line=dict(color='cyan', width=2)
            ))
            fig.update_layout(
                title='Stock Trend of ' + stock,
                xaxis_title='Date',
                yaxis_title='Price ($) '
            )
            st.plotly_chart(fig, use_container_width=True)

            # Simple heuristic for trend feedback
            if stock_df.iloc[500]['close'] > stock_df.iloc[0]['close']:
                original_title = '<b><p style="font-family:Play; color:Cyan; font-size: 20px;">NOTE:<br>Stock is on a solid upward trend. Investing here might be profitable.</p>'
                st.markdown(original_title, unsafe_allow_html=True)
            else:  
                original_title = '<b><p style="font-family:Play; color:Red; font-size: 20px;">NOTE:<br> Stock does not appear to be in a solid uptrend. Better not to invest here; instead, pick different stock.</p>'
                st.markdown(original_title, unsafe_allow_html=True)
    
        # Sidebar Checkbox: Investment Simulation
        st.sidebar.subheader("Enter Your Available Initial Investment Fund")
        invest = st.sidebar.slider('Select a range of values', 1000, 1000000)
        
        if st.sidebar.button("Calculate", key=2):
            # Load Pre-trained Model
            try:
                # Using 'model.pkl' as standardized
                q_table = pkl.load(open('model.pkl', 'rb'))
            except FileNotFoundError:
                st.error("Model file 'model.pkl' not found. Please ensure the model is trained.")
                return

            # Run Simulation
            net_worth = test_stock(stock_df, q_table, invest)
            net_worth = pd.DataFrame(net_worth, columns=['value'])
            
            # Plot Results
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=net_worth.index, 
                y=net_worth['value'],
                mode='lines',
                name='Net_Worth_Trend',
                line=dict(color='cyan', width=2)
            ))
            fig.update_layout(
                title='Change in Portfolio Value Day by Day',
                xaxis_title='Number of Days since Feb 2013 ',
                yaxis_title='Value ($) '
            )
            st.plotly_chart(fig, use_container_width=True)
            
            original_title = '<b><p style="font-family:Play; color:Cyan; font-size: 20px;">NOTE:<br> Increase in your net worth as a result of a model decision.</p>'
            st.markdown(original_title, unsafe_allow_html=True)


if __name__ == '__main__':
    fun()
    # Dummy chart for layout purposes if needed, otherwise optional
    # chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['a', 'b', 'c'])
