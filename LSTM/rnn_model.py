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


def rnn(features, labels, dropout, num_layers, cell_size, dense_units, technicals):
    batch_size = tf.shape(features)[0]
    n_outputs = tf.shape(labels)[1]
    initial_state = tf.zeros([batch_size, num_layers * cell_size])
    
    # RNN
    cells = layers.StackedRNNCells([layers.GRUCell(cell_size) for _ in range(num_layers)])
    rnn_output = layers.RNN(cells, features, initial_state=initial_state) 
    
    # Dropout
    dropout = layers.Dropout(rnn_output[:,-1], keep_prob=1-dropout)
    
    # Dense Layers
    dense_layer = layers.Dense(dropout, dense_units, activation=tf.nn.selu)
    preds = tf.layers.Dense(dense_layer,n_outputs,activation=tf.sigmoid)
    return preds


def lstm(features, labels, dropout, num_layers, cell_size, dense_units, technicals):
    batch_size = tf.shape(features)[0]
    n_outputs = tf.shape(labels)[1]
    initial_state = tf.zeros([batch_size, num_layers * cell_size])
    
    # RNN
    cells = layers.StackedRNNCells([layers.LSTMCell(cell_size) for _ in range(num_layers)])
    rnn_output = layers.RNN(cells, features, initial_state=initial_state) 
    
    # Dropout
    dropout = layers.Dropout(rnn_output[:,-1], keep_prob=1-dropout)
    
    # Dense Layers
    dense_layer = layers.Dense(dropout, dense_units, activation=tf.nn.selu)
    preds = tf.layers.Dense(dense_layer,n_outputs,activation=tf.sigmoid)
    return preds

