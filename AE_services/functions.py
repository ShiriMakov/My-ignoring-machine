from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras import Input, Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import random





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