import os
import sys
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def main():

    print("Beginning program execution...")

    # model = Sequential()

    # model.add(Conv2D(
    #     filters=32,
    #     kernel_size=[2,2],
    #     padding="same",
    #     activation=tf.nn.relu,
    #     data_format='channels_last'
    # ))
    # model.add(MaxPool2D(pool_size=[2,2], strides=2))

    # model.add(Conv2D(
    #     filters=64,
    #     kernel_size=[2,2],
    #     padding="same",
    #     activation=tf.nn.relu
    # ))
    # model.add(MaxPool2D(pool_size=[2,2], strides=2))

    split_path = "data/split_data/"

    train_samples = pickle.load(open(split_path+'train_samples.pickle', 'rb'))
    test_samples = pickle.load(open(split_path+'test_samples.pickle', 'rb'))

    X_train = train_samples[0]
    y_train = train_samples[1]

    X_test = test_samples[0]
    y_test = test_samples[1]

    print("Downloaded input shape: {}".format(X_train.shape))

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = 100

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    model = tf.keras.Sequential([
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(train_dataset, epochs=5)

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)

    predictions = model.predict(X_test)

    print("Predictions:")
    for prediction_probs in predictions[0:10]:
        print(np.argmax(prediction_probs))

    print("Actual Values:")
    for actual in y_test[0:10]:
        print(actual)

if __name__ == "__main__":
    main()
