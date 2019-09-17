import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

from scipy.io.wavfile import read
from scipy.io.wavfile import WavFileWarning
from scipy.fftpack import fft
from scipy.signal import spectrogram

import warnings
warnings.filterwarnings("ignore")

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

def save_spec(wav_data, current_filename, data_root):

    spec_path = data_root + 'spectrograms/'
    if not os.path.exists(spec_path):
        os.mkdir(spec_path)

    plt.figure()
    f, t, Sxx = spectrogram(wav_data, f_s)
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.axis('off')

    output_filename = spec_path + current_filename
    output_filename = output_filename.replace(".wav", ".png")

    if not os.path.exists(output_filename):
        #print("Saving {}...".format(output_filename))
        plt.savefig(output_filename, bbox_inches = 'tight', pad_inches = 0)
    #else:
        #print("File {} already exists...".format(output_filename))

    plt.close()

def iterate_data(data_root):

    data_path = data_root + 'recordings/'
    print("Dataset contains {} overall files\n".format(len(os.listdir(data_path))))

    speakers = [
        'jackson',
        'nicolas',
        'theo'
    ]

    chunk_size = 1024

    print("Saving spectrograms...")
    for wav_path in tqdm(os.listdir(data_path)):
        full_path = data_path + wav_path

        for speaker in speakers:
            if speaker in full_path:
                current_speaker = speaker
        digit = wav_path[0]

        #print("{}".format(full_path))
        rate, wav_data = read(full_path)

        if wav_data.ndim > 1:
            wav_data = wav_data[0,:]

        save_spec(wav_data, wav_path, data_root)

if __name__ == '__main__':
    main()
