import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.io.wavfile import read
from scipy.io.wavfile import WavFileWarning
from scipy.fftpack import fft
from scipy.signal import spectrogram

'''
Files are named in the following format: {digitLabel}_{speakerName}_{index}.wav Example: 7_jackson_32.wav
'''

# Sampling rate in Hz
f_s = 8000

def main():

    print("Beginning program execution...\n")

    # Dummy variables (likely to be cmdline args later)
    data_root = 'data/'

    iterate_data(data_root)

def plot_spec(wav_data, speaker, digit):

    plt.figure(figsize=(10,10))

    plt.title("{} Saying {} (Spectrogram)".format(speaker.capitalize(), digit))

    f, t, Sxx = spectrogram(wav_data, f_s)

    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    plt.show()

def plot_fft(wav_data, speaker, digit):

    plt.figure(figsize=(10,10))

    plt.title("{} Saying {} (Frequency)".format(speaker.capitalize(), digit))

    num_samples = len(wav_data)
    period = 1.0 / f_s
    x_fft = np.linspace(0.0, (1.0 / (2.0 * (period))), num_samples//2)
    y_fft = fft(wav_data)
    plt.plot(
        x_fft,
        2.0 / num_samples * np.abs(y_fft[0:num_samples//2])
    )

    plt.show()

def plot_time(wav_data, speaker, digit):

    plt.figure(figsize=(10,10))

    plt.title("{} Saying {}".format(speaker.capitalize(), digit))
    plt.plot(
        range(len(wav_data)),
        wav_data
    )

    plt.show()

def print_wav_info(wav_data):

    print("Number of samples in wav file: {}".format(len(wav_data)))

def iterate_data(data_root):

    data_path = data_root + 'recordings/'
    print("Dataset contains {} overall files\n".format(len(os.listdir(data_path))))

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

        print("{}".format(full_path))
        rate, wav_data = read(full_path)

        if wav_data.ndim > 1:
            wav_data = wav_data[0,:]

        print_wav_info(wav_data)

        plot_time(wav_data, current_speaker, digit)
        plot_fft(wav_data, current_speaker, digit)
        plot_spec(wav_data, current_speaker, digit)

if __name__ == '__main__':
    main()
