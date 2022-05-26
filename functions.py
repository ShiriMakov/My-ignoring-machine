from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras import Input, Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import random


def add_irrelevant_signal(dataset, dots_num, starting_point=0):
    # dots_num = the number of dots added to the image
    # starting_point = the index of the first pixel to accommodate a dot in the regular-noise image
    dot_ind = np.arange(starting_point, len(dataset[0]), int(np.ceil(len(dataset[0]) / dots_num)))
    dots_num = len(dot_ind)  # update according to what really appears on the image

    regular_noise = np.zeros(dataset[0].shape)
    regular_noise[dot_ind] = 1

    dataset_random_noise = dataset.copy()
    dataset_regular_noise = dataset.copy()

    for i in np.arange(len(dataset)):
        # place n dots randomly
        random_noise = np.zeros(dataset[0].shape)
        random_noise[random.sample(range(0, len(dataset[0])), dots_num)] = 1
        # apply dots onto the digit images
        dataset_random_noise[i] += random_noise
        dataset_regular_noise[i] += regular_noise

    # dataset_random_noise = tf.clip_by_value(dataset_random_noise, clip_value_min=0., clip_value_max=1.)
    # dataset_regular_noise = tf.clip_by_value(dataset_regular_noise, clip_value_min=0., clip_value_max=1.)
    for n, p in enumerate(dataset_regular_noise[0:9]):
        ax1 = plt.subplot(3, 3, n + 1)
        plt.gray()
        ax1.imshow(p.reshape(28, 28))
        ax1.get_xaxis().set_ticks([])

    return dataset_regular_noise, dataset_random_noise


#  ax1.get_yaxis().set_ticks([])

# for n, p in enumerate(x_train_random_noise[0:9]):
#     # add a new subplot iteratively
#     ax2 = plt.subplot(3, 3, n + 1)
#     plt.gray()
#     ax2.imshow(p.reshape(28, 28))
#     ax2.get_xaxis().set_ticks([])
#     ax2.get_yaxis().set_ticks([])



# import math        #import needed modules
# import pyaudio     #sudo apt-get install python-pyaudio
# def create_sound_wave():
#     PyAudio = pyaudio.PyAudio     #initialize pyaudio
#
#     #See https://en.wikipedia.org/wiki/Bit_rate#Audio
#     BITRATE = 14400     #number of frames per second/frameset.
#
#     FREQUENCY = 500     #Hz, waves per second, 261.63=C4-note.
#     LENGTH = 10     #seconds to play sound
#
#     if FREQUENCY > BITRATE:
#         BITRATE = FREQUENCY+100
#
#     NUMBEROFFRAMES = int(BITRATE * LENGTH)
#     RESTFRAMES = NUMBEROFFRAMES % BITRATE
#     WAVEDATA = ''
#
#     #generating wawes
#     for x in xrange(NUMBEROFFRAMES):
#      WAVEDATA = WAVEDATA+chr(int(math.sin(x/((BITRATE/FREQUENCY)/math.pi))*127+128))
#
#     for x in xrange(RESTFRAMES):
#      WAVEDATA = WAVEDATA+chr(128)
#
#     p = PyAudio()
#     stream = p.open(format = p.get_format_from_width(1),
#                     channels = 1,
#                     rate = BITRATE,
#                     output = True)
#
#     stream.write(WAVEDATA)
#     stream.stop_stream()
#     stream.close()
#     p.terminate()
#
