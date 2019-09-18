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

    model = Sequential([
        Flatten(input_shape=(369, 496)),
        Dense(128, activation=tf.nn.relu),
        Dense(10, activation=tf.nn.softmax)
    ])

    split_path = "data/split_data/"

    train_samples = pickle.load(open(split_path+'train_samples.pickle', 'rb'))
    test_samples = pickle.load(open(split_path+'test_samples.pickle', 'rb'))

    X_train, y_train = zip(*train_samples)
    X_test, y_test = zip(*test_samples)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(X_train, y_train, epochs=5)

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)

    predictions = model.predict(X_test)

    print("Predictions:")
    for prediction_probs in predictions[0:10]:
        print(np.argmax(predicition_probs))

    print("Actual Values:")
    for actual in test_y[0:10]:
        print(actual)

if __name__ == "__main__":
    main()
