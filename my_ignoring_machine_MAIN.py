# taken from: https://keras.io/examples/vision/autoencoder/

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import my_ignoring_machine_utils as my_utils
from scipy.stats import mannwhitneyu

from tensorflow.keras import layers, utils
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.metrics import MeanAbsoluteError


# ##########################################################################################################
# Build the autoencoder
do_simple_model = True
do_train_digit_recognition = False

if do_simple_model:
    model_input = layers.Input(shape=(784,))

    # Encoder
    encoded = layers.Dense(16, activation='relu')(model_input)

    # Decoder
    decoded = layers.Dense(784, activation='sigmoid')(encoded)

    # Autoencoder
    autoencoder = Model(inputs=model_input, outputs=decoded)

else:
    model_input = layers.Input(shape=(28, 28, 1))

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(
        model_input)  # padding="same" preserves spatial resolution
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)

    # Decoder
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

    # Autoencoder
    autoencoder = Model(model_input, x)

autoencoder.compile(optimizer="adam",
                    loss="binary_crossentropy",
                    metrics=['accuracy'])

autoencoder.summary()
# save model flow illustration to disc
utils.plot_model(autoencoder, "model_flow_illustration.png", show_shapes=True)


# ##########################################################################################################
# Prepare the data
(x_train, _), (x_test, y_test) = mnist.load_data()

# Normalize and reshape the data
x_train = my_utils.preprocess(x_train, do_simple_model)
x_test = my_utils.preprocess(x_test, do_simple_model)

# Create a copy of the data with added irrelevant information (regular / random)
x_train_regular_noise, x_train_random_noise, regular_noise = my_utils.add_irrelevant_information(x_train, do_simple_model)
x_test_regular_noise, x_test_random_noise, _ = my_utils.add_irrelevant_information(x_test, do_simple_model, regular_noise)

# Display examples from the train data including the added irrelevant information (structured / random noise)
my_utils.display_data(x_train, x_train_regular_noise, x_train_random_noise)


# ##########################################################################################################
# Fit the model according to the training dataset
epoch_num = 15
batch_size = 256
shuffle = True
validation_split = 0.2
# training removing structured irrelevant information
history_regu = autoencoder.fit(x=x_train_regular_noise, y=x_train,
                               epochs=epoch_num,
                               batch_size=batch_size,
                               shuffle=shuffle,
                               validation_split=validation_split)
# training removing random irrelevant information
history_rand = autoencoder.fit(x=x_train_random_noise, y=x_train,
                               epochs=epoch_num,
                               batch_size=batch_size,
                               shuffle=shuffle,
                               validation_split=validation_split)


# ##########################################################################################################
# present examples of the denoising results
predictions_regu = autoencoder.predict(x_test_regular_noise)
predictions_rand = autoencoder.predict(x_test_random_noise)
my_utils.display_results(x_test_regular_noise, predictions_regu, x_test, predictions_rand, x_test_random_noise)


# ##########################################################################################################
# Assess the removal of structured vs random noise

# 1) plot loss and accuracy plots over training epochs
my_utils.plot_metrics(history_regu, history_rand)

# 2) assess the denoising via an MNIST digit-recognizer
# the digit-recognizer used here is by Gregor Koehler (https://nextjournal.com/gkoehler/digit-recognition-with-keras)
# my_utils.run_MNIST_digit_recognizer()

save_dir = os.getcwd()
if do_train_digit_recognition:
    # in case you want to run the model:
    mnist_model = my_utils.run_MNIST_digit_recognizer(save_dir)

model_name = 'keras_mnist.h5'
mnist_model = load_model(os.path.join(save_dir, model_name))

# number of runs to create the distribution of accuracy scores
num_runs = 200

# prepare the labels for testing, using one-hot encoding using keras' numpy-related utilities
n_classes = 10
Y_test = utils.to_categorical(y_test, n_classes)

losses_regu = []
accu_regu = []
losses_rand = []
accu_rand = []
for i in np.arange(num_runs):
    # adding irrelevant information to the images
    x_test_regular_noise, x_test_random_noise, _ = my_utils.add_irrelevant_information(x_test, do_simple_model)
    # testing images with regular 'noise'
    loss_and_metrics = mnist_model.evaluate(x_test_regular_noise, Y_test, verbose=2)
    losses_regu.append(loss_and_metrics[0])
    accu_regu.append(loss_and_metrics[1])
    # testing images with random noise
    loss_and_metrics = mnist_model.evaluate(x_test_random_noise, Y_test, verbose=2)
    losses_rand.append(loss_and_metrics[0])
    accu_rand.append(loss_and_metrics[1])

# plot comparison between regular vs random noise
my_utils.plot_comparison_results(accu_regu, accu_rand)

# Mann-Whitney U test, comparing the categorization accuracy of digit images after denoising regular vs. random noise
stat, p = mannwhitneyu(accu_rand, accu_regu)
print('Statistics=%.3f, p=%.3f' % (stat, p))



