import numpy as np
import pandas as pd
from pandas._libs.missing import NA
import streamlit as st
import time
import plotly.graph_objects as go
import pickle as pkl

##############################################################################################################################
############################ Processing Phase ################################################
# @st.cache(persist=True)
def data_prep(data,name):    
    df=pd.DataFrame(data[data['Name']==name])
    df.dropna(inplace=True)    
    df.reset_index(drop=True,inplace=True)
    # Calculating 5 day and 1 day Moving Average for DF
    df['5day_MA']=df['close'].rolling(5).mean()
    df['1day_MA']=df['close'].rolling(1).mean()
    df['5day_MA'][:4]=0
    #Splitting into train and Test data
    
    return df

# @st.cache(persist=True)
def get_state(long_ma,short_ma,t):
    if short_ma<long_ma:
        if t==1:
            return (0,1) #Cash
        else :
            return (0,0) #Stock
    
    elif short_ma>long_ma:
        if t==1:
            return (1,1) #Cash
        else :
            return (1,0) #Stock


# @st.cache(persist=True)
def trade_t(num_of_stocks,port_value,current_price):
    if num_of_stocks>=0:
        if port_value>current_price:
            return 1
        else :return 0
    else:
        if port_value>current_price:
            return 1
        else :return 0



# @st.cache(persist=True)
def next_act(state,qtable,epsilon,action=3):
    if np.random.rand() < epsilon:
        action=np.random.randint(action)
    else:
        action=np.argmax(qtable[state])
        
        
    return action


# @st.cache(persist=True)
def test_stock(stocks_test,q_table,invest):

    num_stocks=0
    epsilon=0
    net_worth=[invest]
    np.random.seed()

    for dt in range(len(stocks_test)):
        long_ma=stocks_test.iloc[dt]['5day_MA']
        short_ma=stocks_test.iloc[dt]['1day_MA']
        close_price=stocks_test.iloc[dt]['close']
        t=trade_t(num_stocks,net_worth[-1],close_price)
        state=get_state(long_ma,short_ma,t)
        action=next_act(state,q_table,epsilon)

        if action==0:#Buy
            num_stocks+=1
            to_append=net_worth[-1]-close_price
            net_worth.append(np.round(to_append,1))
            
        
        elif action==1:#Sell
            num_stocks-=1
            to_append=net_worth[-1]+close_price
            net_worth.append(np.round(to_append,1))
        
        elif action==2:#hold
            to_append=net_worth[-1]+close_price
            net_worth.append(np.round(to_append,1))
            
        
        
         

      
        try:
            next_state=get_state(stocks_test.iloc[dt+1]['5day_MA'],stocks_test.iloc[dt+1]['1day_MA'],t)
            
        except:
            break

    return net_worth


######################################################################################################################################
############## Designing Phase #############################

def fun():
    #Reading the Dataset
    data=pd.read_csv('all_stocks_5yr.csv')
    names=list(data['Name'].unique())
    names.insert(0,"<Select Names>")

    st.title("Optimizing Stock Trading Strategy With Reinforcement Learning")

    st.sidebar.title("Choose Stock and Investment")

    st.sidebar.subheader("Choose Company Stocks")
    stock=st.sidebar.selectbox("(*select one stock only)",names,index=0)
    stock_df=data_prep(data,stock)
    
 #Sidebar Checkbox Plot The Treand of Data

    if st.sidebar.button("Show Stock Trend",key=1):
        fig=go.Figure()
                  
        fig.add_trace(go.Scatter(x = stock_df['date'], y =stock_df['close'],
                                 mode = 'lines',
                                 name = 'Stock_Trend',line=dict(color='cyan', width=2)))
        fig.update_layout(title='Stock Trend of '+stock,
                xaxis_title='Date',
                yaxis_title='Price ($) ')
        st.plotly_chart(fig, use_container_width=True)

        if stock_df.iloc[500]['close']>stock_df.iloc[0]['close']:
            original_title = '<b><p style="font-family:Play; color:Cyan; font-size: 20px;">NOTE:<br>Stock is on a solid upward trend. Investing here might be profitable.</p>'
            st.markdown(original_title, unsafe_allow_html=True)
        else:  
            original_title = '<b><p style="font-family:Play; color:Red; font-size: 20px;">NOTE:<br> Stock does not appear to be in a solid uptrend. Better not to invest here; instead, pick different stock.</p>'
            st.markdown(original_title, unsafe_allow_html=True)
 
 #Sidebar checkbox Investment value
    st.sidebar.subheader("Enter Your Available Initial Investment Fund")
    invest = st.sidebar.slider('Select a range of values',1000, 1000000)
    if  st.sidebar.button("Calculate", key=2):
        q_table=pkl.load(open('pickl.pkl','rb'))
        net_worth=test_stock(stock_df,q_table,invest)
        net_worth=pd.DataFrame(net_worth,columns=['value'])
        fig=go.Figure()
                  
        fig.add_trace(go.Scatter(x = net_worth.index, y =net_worth['value'],
                                 mode = 'lines',
                                 name = 'Stock_Trend',line=dict(color='cyan', width=2)))
        fig.update_layout(title='Change in Portfolio Value Day by Day',
                   xaxis_title='Number of Days since Feb 2013 ',
                   yaxis_title='Value ($) ')
        st.plotly_chart(fig, use_container_width=True)
        original_title = '<b><p style="font-family:Play; color:Cyan; font-size: 20px;">NOTE:<br> Increase in your net worth as a result of a model decision.</p>'
        st.markdown(original_title, unsafe_allow_html=True)



if __name__=='__main__':
    fun()
    chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c'])
