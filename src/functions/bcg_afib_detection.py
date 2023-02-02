# -*- coding: utf-8 -*-
# BCG AFib Detection Functions

#initializing
from keras import backend as K
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.compat.v1.keras.layers import CuDNNLSTM

#initializing
from keras import backend as K
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.compat.v1.keras.layers import CuDNNLSTM

def create_model(input_shape1, input_shape2, cnn_filters_number=32, activation_function=tf.keras.layers.ReLU(), dropout_ratio=0.2, summary_verbose=True, model_plot=False):
  '''
      Create 1D CNN model

      Parameters:
      input_shape (array): Model input shape
      cnn_filters_number (int): number of filters in first CNN layer.
      activation_function (str, func): activation function
      dropout_ratio (float): ratio of dropout layer, if 0 (default) there will be no dropout layer
      summary_verbose (bool): status of printing summary of the model
      model_plot (bool): status of ploting the model architecture

      Returns: 
      model (tf model): created Tensorflow model
  '''

  inputs1 = keras.Input(shape=input_shape1, name="BCG")
  inputs2 = keras.Input(shape=input_shape2, name="AUTO")
  
  activation_function = tf.keras.layers.ReLU()
  filters_number = 32
  # Branch 1
  CNN1 = layers.Conv1D(filters=filters_number, kernel_size=100, strides=5, name="CNN1")(inputs1) #, activation="gelu"
  CNN1 = layers.BatchNormalization()(CNN1)
  CNN1 = keras.layers.Activation(activation_function)(CNN1)
  if dropout_ratio:
      CNN1 = layers.Dropout(dropout_ratio)(CNN1)

  #****************************************************
  #local feature block
  BlockCNN1 = layers.Conv1D(filters=filters_number*2, kernel_size=3, strides=2, padding='same', name="BlockCNN11")(CNN1)
  BlockCNN1 = layers.BatchNormalization()(BlockCNN1)
  BlockCNN1 = keras.layers.Activation(activation_function)(BlockCNN1)
  BlockCNN1 = layers.Conv1D(filters=filters_number*2, kernel_size=3, strides=1, padding='same', name="BlockCNN12")(BlockCNN1)
  BlockCNN1 = layers.BatchNormalization()(CNN1)

  BlockCNN2 = layers.Conv1D(filters=filters_number*2, kernel_size=1, strides=2, name="BlockCNN21")(CNN1)
  BlockOut1 = layers.Concatenate(name="Merging1")([BlockCNN1, BlockCNN2])
  BlockOut1 = keras.layers.Activation(activation_function)(BlockOut1)
  # CNN11 = layers.Dropout(0.2)(CNN11)

  #local feature block
  BlockCNN3 = layers.Conv1D(filters=filters_number*4, kernel_size=3, strides=2, padding='same', name="BlockCNN31")(BlockOut1)
  BlockCNN3 = layers.BatchNormalization()(BlockCNN3)
  BlockCNN3 = keras.layers.Activation(activation_function)(BlockCNN3)
  BlockCNN3 = layers.Conv1D(filters=filters_number*4, kernel_size=3, strides=1, padding='same', name="BlockCNN32")(BlockCNN3)
  BlockCNN3 = layers.BatchNormalization()(BlockCNN3)

  BlockCNN4 = layers.Conv1D(filters=filters_number*4, kernel_size=1, strides=2, name="BlockCNN41")(BlockOut1)
  BlockOut2 = layers.Concatenate(name="Merging2")([BlockCNN3, BlockCNN4])
  BlockOut2 = keras.layers.Activation(activation_function)(BlockOut2)
  # CNN11 = layers.Dropout(0.2)(CNN11)

  #local feature block
  BlockCNN5 = layers.Conv1D(filters=filters_number*8, kernel_size=3, strides=2, padding='same', name="BlockCNN51")(BlockOut2)
  BlockCNN5 = layers.BatchNormalization()(BlockCNN5)
  BlockCNN5 = keras.layers.Activation(activation_function)(BlockCNN5)
  BlockCNN5 = layers.Conv1D(filters=filters_number*8, kernel_size=3, strides=1, padding='same', name="BlockCNN52")(BlockCNN5)
  BlockCNN5 = layers.BatchNormalization()(BlockCNN5)

  BlockCNN6 = layers.Conv1D(filters=filters_number*8, kernel_size=1, strides=2, name="BlockCNN61")(BlockOut2)
  BlockOut3 = layers.Concatenate(name="Merging3")([BlockCNN5, BlockCNN6])
  BlockOut3 = keras.layers.Activation(activation_function)(BlockOut3)
  # CNN11 = layers.Dropout(0.2)(CNN11)

  #local feature block
  BlockCNN7 = layers.Conv1D(filters=filters_number*16, kernel_size=3, strides=2, padding='same', name="BlockCNN71")(BlockOut3)
  BlockCNN7 = layers.BatchNormalization()(BlockCNN7)
  BlockCNN7 = keras.layers.Activation(activation_function)(BlockCNN7)
  BlockCNN7 = layers.Conv1D(filters=filters_number*16, kernel_size=3, strides=1, padding='same', name="BlockCNN72")(BlockCNN7)
  BlockCNN7 = layers.BatchNormalization()(BlockCNN7)

  BlockCNN8 = layers.Conv1D(filters=filters_number*16, kernel_size=1, strides=2, name="BlockCNN81")(BlockOut3)
  BlockOut4 = layers.Concatenate(name="Merging4")([BlockCNN7, BlockCNN8])
  BlockOut4 = keras.layers.Activation(activation_function)(BlockOut4)
  # CNN11 = layers.Dropout(0.2)(CNN11)
  #****************************************************
  avr1 = tf.keras.layers.GlobalAveragePooling1D()(BlockOut4)
  ###################################################################################################
  # Branch 2
  CNN2 = layers.Conv1D(filters=filters_number, kernel_size=100, strides=5, name="CN1")(inputs2) #, activation="gelu"
  CNN2 = layers.BatchNormalization()(CNN2)
  CNN2 = keras.layers.Activation(activation_function)(CNN2)
  if dropout_ratio:
      CNN2 = layers.Dropout(dropout_ratio)(CNN2)
  
  #****************************************************
  #local feature block
  Block2CNN1 = layers.Conv1D(filters=filters_number*8, kernel_size=3, strides=2, padding='same', name="Block2CNN11")(CNN2)
  Block2CNN1 = layers.BatchNormalization()(Block2CNN1)
  Block2CNN1 = keras.layers.Activation(activation_function)(Block2CNN1)
  Block2CNN1 = layers.Conv1D(filters=filters_number*8, kernel_size=3, strides=1, padding='same', name="Block2CNN12")(Block2CNN1)
  Block2CNN1 = layers.BatchNormalization()(Block2CNN1)

  Block2CNN2 = layers.Conv1D(filters=filters_number*8, kernel_size=1, strides=2, name="Block2CNN21")(CNN2)
  Block2Out1 = layers.Concatenate(name="Merging5")([Block2CNN1, Block2CNN2])
  Block2Out1 = keras.layers.Activation(activation_function)(Block2Out1)
  # CNN11 = layers.Dropout(0.2)(CNN11)

  #local feature block
  Block2CNN3 = layers.Conv1D(filters=filters_number*16, kernel_size=3, strides=2, padding='same', name="Block2CNN31")(Block2Out1)
  Block2CNN3 = layers.BatchNormalization()(Block2CNN3)
  Block2CNN3 = keras.layers.Activation(activation_function)(Block2CNN3)
  Block2CNN3 = layers.Conv1D(filters=filters_number*16, kernel_size=3, strides=1, padding='same', name="Block2CNN32")(Block2CNN3)
  Block2CNN3 = layers.BatchNormalization()(Block2CNN3)

  Block2CNN4 = layers.Conv1D(filters=filters_number*16, kernel_size=1, strides=2, name="Block2CNN41")(Block2Out1)
  Block2Out2 = layers.Concatenate(name="Merging6")([Block2CNN3, Block2CNN4])
  Block2Out2 = keras.layers.Activation(activation_function)(Block2Out2)
  # CNN11 = layers.Dropout(0.2)(CNN11)
  #*****************************************************
  
  avr2 = tf.keras.layers.GlobalAveragePooling1D()(Block2Out2)

  con = layers.Concatenate(name="Merging7")([avr1, avr2])

  dense1 = layers.Dense(256, name="FC1")(con)
  dense1 = keras.layers.Activation(activation_function)(dense1)
  if dropout_ratio:
      dense1 = layers.Dropout(dropout_ratio)(dense1)

  # dense1 = layers.Dense(64, name="FC2")(dense1)
  # dense1 = keras.layers.Activation(tf.keras.activations.gelu)(dense1)
  # dense1 = layers.Dropout(0.2)(dense1)

  dense1 = layers.Dense(32, name="FC3")(dense1)
  dense1 = keras.layers.Activation(tf.keras.activations.gelu)(dense1)
  if dropout_ratio:
      dense1 = layers.Dropout(dropout_ratio)(dense1)

  outputs = layers.Dense(1, activation="sigmoid", name="Output")(dense1)

  # model = keras.Model(inputs=[inputs1, inputs2], outputs=outputs, name="LSTM_model")
  model = keras.Model(inputs=[inputs1, inputs2], outputs=outputs, name="LSTM_model")

  if summary_verbose:
      model.summary()

  if model_plot:
      tf.keras.utils.plot_model(model)

  return model
#end create_model#end create_model

def my_metric_bcg_model(y_true, y_pred):
    """
    Metric for training phase of model that return square of summation of precision and recall

    Parameters:
    y_true (tensor): ground truth labels
    y_pred (tensor): predicted labels

    Returns: 
    square of summation of precision and recall
    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_keras = true_positives / (predicted_positives + K.epsilon())

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_keras = true_positives / (possible_positives + K.epsilon())

    return (precision_keras+recall_keras)*(precision_keras+recall_keras)
#end my_metric_bcg_model

def get_lr_per_epoch(scheduler, num_epoch):
    lr_per_epoch = []
    for epoch in range(num_epoch):
        lr_per_epoch.append(scheduler.get_epoch_values(epoch))
    return lr_per_epoch
#end get_lr_per_epoch

def cosine_learning_rate_scheduler(num_epoch=200, restart_number=30, start_lr=1e-3, lr_min=1e-5, cycle_limit=4, cycle_mul=2, cycle_decay=0.5, plot_status=True):
    # Configuring Learning Rate
    from timm.scheduler.cosine_lr import CosineLRScheduler
    from matplotlib import pyplot as plt
    from timm import create_model as create_model_tim
    from timm.optim import create_optimizer
    from types import SimpleNamespace

    model_tim = create_model_tim('resnet18')

    args = SimpleNamespace()
    args.weight_decay = 0
    args.lr = start_lr
    args.opt = 'adam' 
    args.momentum = 0.09

    optimizer = create_optimizer(args, model_tim)

    scheduler = CosineLRScheduler(optimizer, t_initial=restart_number, lr_min=lr_min, warmup_t=0, warmup_lr_init=1e-3, cycle_limit=cycle_limit, cycle_decay=cycle_decay, cycle_mul=cycle_mul)
    lr_per_epoch = get_lr_per_epoch(scheduler, num_epoch)

    if plot_status:
        fig1 = plt.figure(figsize = [15,10])
        plt.title('Cosine Learning Rate')
        plt.plot([i for i in range(num_epoch)], lr_per_epoch);
    #end

    return scheduler
#end cosine_learning_rate_scheduler

def keras_scheduler(epoch):
    return scheduler.get_epoch_values(epoch)[0]
#end keras_scheduler