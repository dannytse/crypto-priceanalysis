import json
import requests
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
# import torch
# import torch.nn as nn
# import torch

#k-fold constants
folds=5
min_train_size=0.5

#preprocess constants
window_len = 22
test_val_size = 0.15

#model constants
epochs=5
loss='mse'
optimizer = 'adam'
dropout=0.33
num_layers=2
cell_size=32
dense_units=95
technicals=False

def get_data():
    df = pd.read_csv('/Users/timwu0/Documents/CS329P/afterhours_crypto/preprocessing/crypto_data.csv')
    coins = df['coin'].unique()
    print(coins)

    #convert coin to one-hot vectors
    for coin in coins:
        df[coin] = df['coin'] == coin
    return coin
    

def main():
    # Replace with your own path
        

    train_data = pd.DataFrame(columns=df.columns)
    test_data = pd.DataFrame(columns=df.columns)
    for coin in coins:
        df_coin = df.loc[df['coin'] == coin]
        split_row_val = len(df_coin) - int(2 * test_val_size * len(df_coin))
        split_row_test = len(df_coin) - int(test_val_size * len(df_coin))
        
        train_data = pd.concat([train_data, df_coin.iloc[:split_row_val]]) 
        val_data = pd.concat([train_data, df_coin.iloc[split_row_val:split_row_test]]) 
        test_data = pd.concat([test_data, df_coin.iloc[split_row_test:]])
        # print(train_data.tail())
    
    # print(train_data.columns)
    mean_p = train_data['p'].mean()
    std_p = train_data['p'].std()
    train_data['p'] = ((train_data['p']-mean_p)/std_p)#.round(1)
    val_data['p'] = ((val_data['p']-mean_p)/std_p)#.round(1)
    test_data['p'] = ((test_data['p']-mean_p)/std_p)#.round(1)
    

    return train_data.drop(['coin'], axis=1), val_data.drop(['coin'], axis=1), test_data.drop(['coin'], axis=1)
train, val, test = train_test_split(df, test_val_size=0.15)
print(train, val, test)
    