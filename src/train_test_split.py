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

        print("Building dataset...")
        for path in tqdm(os.listdir(spec_path)):
            img = cv2.imread(spec_path + path, 0)
            X.append(img)
            y.append(path[0])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_portion/100.0,
            random_state=42
        )

        train_samples = zip(X_train, y_train)
        test_samples  = zip(X_test, y_test)

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
