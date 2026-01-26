"""
Project: Optimizing Stock Trading Strategy With Reinforcement Learning
Authors: Amey Thakur & Mega Satish
Reference: https://github.com/Amey-Thakur/OPTIMIZING-STOCK-TRADING-STRATEGY-WITH-REINFORCEMENT-LEARNING
License: MIT

Description:
This script implements the training phase of the Reinforcement Learning agent (Q-Learning).
It preprocesses historical stock data, defines the market environment as a set of states
based on Moving Average crossovers, and iteratively updates a Q-Table to learn optimal
trading actions (Buy, Sell, Hold) that maximize portfolio returns.
"""

import pandas as pd
import numpy as np
import pickle as pkl
import os

# ==========================================
# 1. Data Preprocessing
# ==========================================
def data_prep(data, name):
    """
    Preprocesses the stock data for a specific company.
    
    Args:
        data (pd.DataFrame): The complete dataset containing all stocks.
        name (str): The ticker symbol of the stock to filter (e.g., 'AAPL').
        
    Returns:
        tuple: (train_df, test_df) - The split training and testing datasets.
        
    Methodology:
    - Filters data by stock name.
    - Computes Technical Indicators: 5-day and 1-day Moving Averages (MA).
        - 5-day MA represents the short-term trend baseline.
        - 1-day MA represents the immediate price action.
    - The interaction between these two MAs serves as the primary signal for state determination.
    """
    df = pd.DataFrame(data[data['Name'] == name])
    df.dropna(inplace=True)
    df.drop(['high', 'low', 'volume', 'Name'], axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Calculating Moving Averages used for State Definition
    df['5day_MA'] = df['close'].rolling(5).mean()
    df['1day_MA'] = df['close'].rolling(1).mean()
    
    # Initialize first few rows where rolling mean is NaN
    df.loc[:4, '5day_MA'] = 0
    
    # Splitting into Train (80%) and Test (20%) sets
    split_idx = int(len(df) * 0.8)
    train_df = df[:split_idx]
    test_df = df[split_idx:].reset_index(drop=True)
    
    return train_df, test_df

# ==========================================
# 2. Environment & State Definitions
# ==========================================
def get_state(long_ma, short_ma, t):
    """
    Discretizes continuous market data into a finite set of states.
    
    The state space is defined by a tuple (Trend_Signal, Holding_Status).
    
    1. Trend_Signal:
       - 0: short_ma < long_ma (Bearish/Downtrend)
       - 1: short_ma > long_ma (Bullish/Uptrend)
       
    2. Holding_Status (t):
       - 0: Currently holding stock
       - 1: Currently holding cash (no stock)
       
    Returns:
        tuple: (trend, holding_status) representing the current environment state.
    """
    if short_ma < long_ma:
        if t == 1:
            return (0, 1) # Bearish Trend, Holding Cash
        else:
            return (0, 0) # Bearish Trend, Holding Stock
            
    elif short_ma > long_ma:
        if t == 1:
            return (1, 1) # Bullish Trend, Holding Cash
        else:
            return (1, 0) # Bullish Trend, Holding Stock
            
    # Default case (should rarely be hit with floats)
    return (0, 1)

def trade_t(num_of_stocks, port_value, current_price):
    """
    Determines the holding capability of the agent.
    
    Returns:
        int: 1 if the agent has capital to buy (Cash), 0 if fully invested (Stock).
    """
    # Simply mapping: if we have stocks or cash value > current price, we can 'technically' buy/hold
    # But in this simplified binary state (All-in or All-out), we track logical status.
    # Here, we simplify:
    if num_of_stocks > 0:
        return 0 # User holds stock
    else:
        if port_value > current_price:
            return 1 # User holds cash and can afford stock
        else: 
            return 0 # User is broke/cannot buy

# ==========================================
# 3. Q-Learning Agent Logic
# ==========================================
def next_act(state, qtable, epsilon, action_space=3):
    """
    Selects the next action using the Epsilon-Greedy Policy.
    
    Args:
        state (tuple): The current state of the environment.
        qtable (np.array): The Q-Table storing action-values.
        epsilon (float): Exploration rate (probability of random action).
        
    Returns:
        int: The selected action index.
            0: Buy
            1: Sell
            2: Hold
    """
    if np.random.rand() < epsilon:
        # Exploration: Random action
        action = np.random.randint(action_space)
    else:
        # Exploitation: Best known action from Q-Table
        action = np.argmax(qtable[state])
        
    return action

def get_reward(state, action, current_close, past_close, buy_history):
    """
    Calculates the immediate reward for a given state-action pair.
    
    The Reward Function is crucial for guiding the agent:
    - Penalize invalid moves (e.g., Buying when already holding).
    - Reward profit generation (Selling higher than bought).
    - Reward capital preservation (Holding during downturns).
    """
    if state == (0, 0) or state == (1, 0): # State: Holding Stock
        if action == 0: # Try to Buy again
            return -1000 # Heavy Penalty for illegal move
        elif action == 1: # Sell
            return (current_close - buy_history) # Reward is the realized PnL
        elif action == 2: # Hold
            return (current_close - past_close) # Reward is the unrealized daily change
    
    elif state == (0, 1) or state == (1, 1): # State: Holding Cash
        if action == 0: # Buy
            return 0 # Neutral reward for entering position
        elif action == 1: # Try to Sell again
            return -1000 # Heavy Penalty for illegal move
        elif action == 2: # Hold (Wait)
            return (current_close - past_close) # Opportunity cost/benefit tracking

    return 0

# ==========================================
# 4. Main Training Loop
# ==========================================
def train_model():
    print("Initializing Training Process...")
    
    # 4.1 Initialize Q-Table
    # Dimensions: 2 (Trend States) x 2 (Holding States) x 3 (Actions)
    env_rows = 2
    env_cols = 2
    n_action = 3
    q_table = np.zeros((env_rows, env_cols, n_action))
    
    # 4.2 Load Data
    try:
        stocks = pd.read_csv('all_stocks_5yr.csv')
        # We train primarily on AAPL as the representative asset for this strategy
        stocks_train, _ = data_prep(stocks, 'AAPL')
    except FileNotFoundError:
        print("Error: 'all_stocks_5yr.csv' not found.")
        return

    # 4.3 Hyperparameters
    episodes = 100       # Number of times to iterate over the dataset
    epsilon = 1.0        # Initial Exploration Rate (100% random)
    alpha = 0.05         # Learning Rate (Impact of new information)
    gamma = 0.15         # Discount Factor (Importance of future rewards)
    
    print(f"Starting Training for {episodes} episodes...")
    
    for i in range(episodes):
        # Reset Episode Variables
        port_value = 1000
        num_stocks = 0
        buy_history = 0
        net_worth = [1000]
        
        # Iterate over the time-series
        for dt in range(len(stocks_train)):
            long_ma = stocks_train.iloc[dt]['5day_MA']
            short_ma = stocks_train.iloc[dt]['1day_MA']
            close_price = stocks_train.iloc[dt]['close']
            
            # Get Previous Close for Reward Calc
            if dt > 0:
                past_close = stocks_train.iloc[dt-1]['close']
            else:
                past_close = close_price
                
            # Determine Current State
            t = trade_t(num_stocks, net_worth[-1], close_price)
            state = get_state(long_ma, short_ma, t)
            
            # Select Action
            action = next_act(state, q_table, epsilon)
            
            # Execute Action & Update Portfolio Logic
            if action == 0: # Buy
                num_stocks += 1
                buy_history = close_price
                net_worth.append(np.round(net_worth[-1] - close_price, 1))
                r = 0 # Reward calculated later if needed, mostly 0 for entry
            
            elif action == 1: # Sell
                num_stocks -= 1
                net_worth.append(np.round(net_worth[-1] + close_price, 1))
                # buy_history handled in reward
            
            elif action == 2: # Hold
                net_worth.append(np.round(net_worth[-1] + close_price, 1)) # Simplified tracking
            
            # Compute Reward
            r = get_reward(state, action, close_price, past_close, buy_history)
            
            # Observe Next State
            try:
                next_long = stocks_train.iloc[dt+1]['5day_MA']
                next_short = stocks_train.iloc[dt+1]['1day_MA']
                next_state = get_state(next_long, next_short, t)
            except IndexError:
                # End of data
                break
                
            # Update Q-Value using Bellman Equation
            # Q(s,a) = (1-alpha) * Q(s,a) + alpha * (reward + gamma * max(Q(s', a')))
            q_table[state][action] = (1. - alpha) * q_table[state][action] + alpha * (r + gamma * np.max(q_table[next_state]))
        
        # Decay Epsilon to reduce exploration over time
        if (epsilon - 0.01) > 0.15:
            epsilon -= 0.01
            
        if (i + 1) % 10 == 0:
            print(f"Episode {i+1}/{episodes} complete. Epsilon: {epsilon:.2f}")

    print("Training Complete.")
    
    # 4.4 Save the Trained Model
    with open('model.pkl', 'wb') as f:
        pkl.dump(q_table, f)
    print("Model saved to 'model.pkl'.")

if __name__ == "__main__":
    train_model()
