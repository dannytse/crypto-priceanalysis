import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as layers
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error


def lstm(features, labels, dropout, neurons, dense_units, technicals):
    '''
    batch_size = tf.shape(features)[0]
    n_outputs = tf.shape(labels)[1]
    initial_state = tf.zeros([batch_size, num_layers * cell_size])
    
    # RNN
    cells = layers.StackedRNNCells([layers.GRUCell(cell_size) for _ in range(num_layers)])
    rnn_layer = layers.RNN(cells)
    rnn_output = rnn_layer(features, initial_state=initial_state) 
    
    # Dropout
    dropout = layers.Dropout(rate=dropout)
    dropout_output = dropout(rnn_output)
    
    # Dense Layers
    dense_layer = layers.Dense(dense_units, activation=tf.nn.selu)
    dense_layer_output = dense_layer(dropout_output)

    final = layers.Dense(n_outputs,activation=tf.sigmoid)
    final_output = final(dense_layer_output)
    '''
    n_outputs = tf.shape(labels)[1]
    # cells = layers.StackedRNNCells([layers.LSTMCell(cell_size) for _ in range(num_layers)])
    model = tf.keras.Sequential()
    model.add(layers.LSTM(neurons, input_shape=(features.shape[1], features.shape[2])))
    # model.add(layers.LSTM(neurons, input_shape=(22, 27)))
    model.add(layers.Dropout(rate=dropout))
    model.add(layers.Activation(activation=tf.sigmoid))
    model.add(layers.Dense(dense_units, activation=tf.nn.selu))
    model.add(layers.Dense(n_outputs,activation=tf.sigmoid))
    
    model.compile(optimizer='adam', loss='mse')
    return model

'''
def lstm(features, labels, dropout, num_layers, cell_size, dense_units, technicals):
    batch_size = tf.shape(features)[0]
    n_outputs = tf.shape(labels)[1]
    initial_state = tf.zeros([batch_size, num_layers * cell_size])
    
    # RNN
    cells = layers.StackedRNNCells([layers.LSTMCell(cell_size) for _ in range(num_layers)])
    rnn_output = layers.RNN(cells, features)(initial_state=initial_state) 
    
    # Dropout
    dropout = layers.Dropout(rnn_output[:,-1], keep_prob=1-dropout)
    
    # Dense Layers
    dense_layer = layers.Dense(dropout, dense_units, activation=tf.nn.selu)
    preds = tf.layers.Dense(dense_layer,n_outputs,activation=tf.sigmoid)
    return preds
'''

