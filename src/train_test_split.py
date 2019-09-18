import os
import sys
import pickle

import cv2
import PIL

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.model_selection import train_test_split

def main():

    print("Beginning program execution...")

    test_portion = 20
    spec_path = "data/spectrograms/"

    split_path = "data/split_data/"

    if not os.path.exists(split_path):
        os.mkdir(split_path)

        X = []
        y = []

        print("Building image dataset...")
        for path in tqdm(os.listdir(spec_path)):
            img = cv2.imread(spec_path + path, 0)
            img = np.expand_dims(img, 3)
            X.append(img)
            y.append(np.float32(path[0]))

        X = np.array(X)
        y = np.array(y)

        print("Normalizing dataset...")
        np.divide(X, 255.0)

        print("Splitting data into train and test datasets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_portion/100.0,
            random_state=42
        )

        print("X shape: {}".format(X.shape))
        print("y shape: {}".format(y.shape))

        train_samples = (X_train, y_train)
        test_samples  = (X_test, y_test)

        print("Saving dataset to filesystem...")
        pickle.dump(train_samples, open(split_path+'train_samples.pickle', 'wb'))
        pickle.dump(test_samples, open(split_path+'test_samples.pickle', 'wb'))
    else:
        print("Found previously created dataset...")
        print("Exiting...")

        # EXAMPLE FOR LOADING
        # train_samples = pickle.load(open(split_path+'train_samples.pickle', 'rb'))
        # test_samples = pickle.load(open(split_path+'test_samples.pickle', 'rb'))

        # X_train, y_train = zip(*train_samples)
        # X_test, y_test = zip(*test_samples)

if __name__ == "__main__":
    main()
