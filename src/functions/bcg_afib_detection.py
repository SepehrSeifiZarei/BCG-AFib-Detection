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
  CNN11 = keras.layers.Activation(activation_function)(CNN1)
  if dropout_ratio:
      CNN11 = layers.Dropout(dropout_ratio)(CNN11)

  #****************************************************
  #local feature block
  CNN1 = layers.Conv1D(filters=filters_number*2, kernel_size=3, strides=2, padding='same', name="CNN2")(CNN11)
  CNN1 = layers.BatchNormalization()(CNN1)
  CNN1 = keras.layers.Activation(activation_function)(CNN1)
  CNN1 = layers.Conv1D(filters=filters_number*2, kernel_size=3, strides=1, padding='same', name="CNN3")(CNN1)
  CNN12 = layers.BatchNormalization()(CNN1)

  CNN22 = layers.Conv1D(filters=filters_number*2, kernel_size=1, strides=2, name="CNN4")(CNN11)
  CNN11 = layers.Concatenate(name="Merging")([CNN12, CNN22])
  CNN11 = keras.layers.Activation(activation_function)(CNN1)
  # CNN11 = layers.Dropout(0.2)(CNN11)

  #local feature block
  CNN1 = layers.Conv1D(filters=filters_number*4, kernel_size=3, strides=2, padding='same', name="CNN5")(CNN11)
  CNN1 = layers.BatchNormalization()(CNN1)
  CNN1 = keras.layers.Activation(activation_function)(CNN1)
  CNN1 = layers.Conv1D(filters=filters_number*4, kernel_size=3, strides=1, padding='same', name="CNN6")(CNN1)
  CNN12 = layers.BatchNormalization()(CNN1)

  CNN22 = layers.Conv1D(filters=filters_number*4, kernel_size=1, strides=2, name="CNN7")(CNN11)
  CNN11 = layers.Concatenate(name="Merging")([CNN12, CNN22])
  CNN11 = keras.layers.Activation(activation_function)(CNN1)
  # CNN11 = layers.Dropout(0.2)(CNN11)

  #local feature block
  CNN1 = layers.Conv1D(filters=filters_number*8, kernel_size=3, strides=2, padding='same', name="CNN8")(CNN11)
  CNN1 = layers.BatchNormalization()(CNN1)
  CNN1 = keras.layers.Activation(activation_function)(CNN1)
  CNN1 = layers.Conv1D(filters=filters_number*8, kernel_size=3, strides=1, padding='same', name="CNN9")(CNN1)
  CNN12 = layers.BatchNormalization()(CNN1)

  CNN22 = layers.Conv1D(filters=filters_number*8, kernel_size=1, strides=2, name="CNN10")(CNN11)
  CNN11 = layers.Concatenate(name="Merging")([CNN12, CNN22])
  CNN11 = keras.layers.Activation(activation_function)(CNN1)
  # CNN11 = layers.Dropout(0.2)(CNN11)

  #local feature block
  CNN1 = layers.Conv1D(filters=filters_number*16, kernel_size=3, strides=2, padding='same', name="CNN11")(CNN11)
  CNN1 = layers.BatchNormalization()(CNN1)
  CNN1 = keras.layers.Activation(activation_function)(CNN1)
  CNN1 = layers.Conv1D(filters=filters_number*16, kernel_size=3, strides=1, padding='same', name="CNN12")(CNN1)
  CNN12 = layers.BatchNormalization()(CNN1)

  CNN22 = layers.Conv1D(filters=filters_number*16, kernel_size=1, strides=2, name="CNN13")(CNN11)
  CNN11 = layers.Concatenate(name="Merging")([CNN12, CNN22])
  CNN11 = keras.layers.Activation(activation_function)(CNN1)
  # CNN11 = layers.Dropout(0.2)(CNN11)
  #****************************************************
  avr1 = tf.keras.layers.GlobalAveragePooling1D()(CNN11)

  # Branch 2
  CNN2 = layers.Conv1D(filters=filters_number, kernel_size=100, strides=5, name="CN1")(inputs2) #, activation="gelu"
  CNN2 = layers.BatchNormalization()(CNN2)
  CNN21 = keras.layers.Activation(activation_function)(CNN2)
  if dropout_ratio:
      CNN21 = layers.Dropout(dropout_ratio)(CNN21)
  
  #****************************************************
  #local feature block
  CNN2 = layers.Conv1D(filters=filters_number*8, kernel_size=3, strides=2, padding='same', name="CN2")(CNN21)
  CNN2 = layers.BatchNormalization()(CNN2)
  CNN2 = keras.layers.Activation(activation_function)(CNN2)
  CNN2 = layers.Conv1D(filters=filters_number*8, kernel_size=3, strides=1, padding='same', name="CN3")(CNN2)
  CNN22 = layers.BatchNormalization()(CNN2)

  CNN23 = layers.Conv1D(filters=filters_number*8, kernel_size=1, strides=2, name="CN4")(CNN21)
  CNN21 = layers.Concatenate(name="Merging")([CNN23, CNN22])
  CNN21 = keras.layers.Activation(activation_function)(CNN2)
  # CNN11 = layers.Dropout(0.2)(CNN11)

  #local feature block
  CNN2 = layers.Conv1D(filters=filters_number*16, kernel_size=3, strides=2, padding='same', name="CN5")(CNN21)
  CNN2 = layers.BatchNormalization()(CNN2)
  CNN2 = keras.layers.Activation(activation_function)(CNN2)
  CNN2 = layers.Conv1D(filters=filters_number*16, kernel_size=3, strides=1, padding='same', name="CN6")(CNN2)
  CNN22 = layers.BatchNormalization()(CNN2)

  CNN23 = layers.Conv1D(filters=filters_number*16, kernel_size=1, strides=2, name="CN7")(CNN21)
  CNN21 = layers.Concatenate(name="Merging")([CNN23, CNN22])
  CNN21 = keras.layers.Activation(activation_function)(CNN2)
  # CNN11 = layers.Dropout(0.2)(CNN11)
  #*****************************************************
  
  avr2 = tf.keras.layers.GlobalAveragePooling1D()(CNN21)

  con = layers.Concatenate(name="Merging")([avr1, avr2])

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