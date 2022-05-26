from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras import Input, Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import functions as f
import random


# provide the number of dimensions that will decide how much the input will be compressed
encoding_dim = 15
input_img = Input(shape=(784,))
# encoded representation of input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# decoded representation of code
decoded = Dense(784, activation='sigmoid')(encoded)
# Model which take input image and shows decoded images
autoencoder = Model(input_img, decoded)

# build the encoder model and decoder model separately so that we can easily differentiate between the input and output
# This model shows encoded images
encoder = Model(input_img, encoded)
# Creating a decoder model
encoded_input = Input(shape=(encoding_dim,))
# last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))


# compile the model with the ADAM optimizer and cross-entropy loss function fitment
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


# load the data
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# add noise (both regular and random)
n = random.randrange(60, 120)  # number of dots added to each image
start_point = random.randrange(0, 10)
x_train_regular_noise, x_train_random_noise = f.add_irrelevant_signal(x_train, n, start_point)
x_test_regular_noise, x_test_random_noise = f.add_irrelevant_signal(x_test, n, start_point)



# ############################################################################################################################
# train
autoencoder.fit(x_train_regular_noise, x_train,
                epochs=15,
                batch_size=256,
                shuffle=True,
                validation_split=0.2)

# provide new noisy input for testing the reconstruction
encoded_img = encoder.predict(x_test_regular_noise)  # MUST UNDERSTAND WHY NOT USE autoencoder HERE!!!!!
decoded_img = decoder.predict(encoded_img)  # AND ALSO HERE!!!!!
plt.figure(figsize=(6, 4))
for i in range(3):
    # Display original
    ax = plt.subplot(2, 3, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Display reconstruction
    ax = plt.subplot(2, 3, i + 1 + 3)
    plt.imshow(decoded_img[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


