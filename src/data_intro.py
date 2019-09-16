import os
import sys
import wave

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():


    print("Beginning program execution...")

    # Dummy variables (likely to be cmdline args later)
    data_root = 'data/'

    data_path = data_root + 'recordings/'

    print("Dataset contains {} overall files".format(len(os.listdir(data_path))))


if __name__ == '__main__':
    main()
