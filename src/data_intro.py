import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.io.wavfile import read

'''
Files are named in the following format: {digitLabel}_{speakerName}_{index}.wav Example: 7_jackson_32.wav
'''

# Sampling rate in Hz
f_s = 8000

def main():

    print("Beginning program execution...\n")

    # Dummy variables (likely to be cmdline args later)
    data_root = 'data/'

    data_path = data_root + 'recordings/'
    print("Dataset contains {} overall files\n".format(len(os.listdir(data_path))))

    iterate_data(data_path)

def plot_waveform(wav_data, speaker, digit):

    plt.figure(figsize=(10,10))

    plt.title("Speaker: {} Digit: {}".format(speaker.capitalize(), digit))
    plt.plot(
        range(len((wav_data))),
        wav_data
    )

    plt.show()

def print_wav_info(wav_data):

    print("Number of samples in wav file: {}".format(len(wav_data)))

def iterate_data(data_path):

    speakers = [
        'jackson',
        'nicolas',
        'theo'
    ]

    chunk_size = 1024

    for wav_path in os.listdir(data_path):
        full_path = data_path + wav_path

        for speaker in speakers:
            if speaker in full_path:
                current_speaker = speaker
        digit = wav_path[0]

        print("Opening {}".format(full_path))
        rate, wav_data = read(full_path)

        print_wav_info(wav_data)
        plot_waveform(wav_data, current_speaker, digit)

        break

if __name__ == '__main__':
    main()
