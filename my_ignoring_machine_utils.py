# functions in service of the MAIN script 'My ignoring machine'

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import seaborn as sns
import pandas as pd


from tensorflow.keras import layers, utils
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.metrics import MeanAbsoluteError


# from tensorflow.keras.layers.core import Dense, Dropout, Activation


def preprocess(array, do_simple_model):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """

    array = array.astype("float32") / 255.0
    if do_simple_model:
        array = np.reshape(array, (len(array), np.prod(array.shape[1:])))
    else:
        array = np.reshape(array, (len(array), 28, 28, 1))

    return array


def add_irrelevant_information(dataset, do_simple_model, regular_noise=[0]):
    """
    Adds regular / random signals to each image in the supplied array.
    """
    # reshape for convenience
    dataset = np.reshape(dataset, (len(dataset), np.prod(dataset.shape[1:])))

    # randomize the dots number and layout
    dots_num = random.randrange(60, 120)  # number of dots added to each image
    starting_point = random.randrange(0, 10)  # randomize startign location
    dot_ind = np.arange(starting_point, len(dataset[0]), int(np.ceil(len(dataset[0]) / dots_num)))
    dots_num = len(dot_ind)  # update according to what really appears on the image

    # reshape data to add noise conveniently
    dataset.reshape((len(dataset), np.prod(dataset.shape[1:])))

    # initiate noise components
    dataset_random_noise = dataset.copy()
    dataset_regular_noise = dataset.copy()

    # add noise
    if len(regular_noise) == 1:  # initiate it, in case it was not pre-defined when calling the function
        regular_noise = np.zeros(dataset[0].shape)
        regular_noise[dot_ind] = 1
    for i in np.arange(len(dataset)):
        # place n dots randomly
        random_noise = np.zeros(dataset[0].shape)
        random_noise[random.sample(range(0, len(dataset[0])), dots_num)] = 1
        # apply dots onto the digit images
        dataset_random_noise[i] += random_noise
        dataset_regular_noise[i] += regular_noise

    # dataset_random_noise = np.clip(dataset_random_noise, 0., 1.)
    # dataset_regular_noise = np.clip(dataset_regular_noise, 0., 1.)

    if not do_simple_model:
        dataset_random_noise = np.reshape(dataset_random_noise, (len(dataset_random_noise), 28, 28, 1))
        dataset_regular_noise = np.reshape(dataset_regular_noise, (len(dataset_regular_noise), 28, 28, 1))

    return dataset_regular_noise, dataset_random_noise, regular_noise


def display_data(array1, array2, array3):
    """
    Displays images from each one of the supplied arrays.
    """

    n = 6

    indices = np.arange(100, 106)  # np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]
    images3 = array3[indices, :]

    plt.figure(figsize=(12, 4))
    for i, (image1, image2, image3) in enumerate(zip(images1, images2, images3)):
        ax = plt.subplot(3, n, i + 1)
        plt.suptitle('Example images with added irrelevant information (structured / random)')
        plt.imshow(image1.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(image2.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(image3.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


def plot_metrics(history_regu, history_rand):
    """
    Displays loss and accuracy plots over training epochs
    """

    plt.figure(figsize=(9, 10))

    ax = plt.subplot(2, 2, 1)
    ax.plot(history_regu.history['accuracy'], 'b')
    ax.plot(history_regu.history['val_accuracy'], 'c')
    ax.plot(history_rand.history['accuracy'], 'r')
    ax.plot(history_rand.history['val_accuracy'], 'y')
    ax.set_title('Denoising accuracy')
    ax.set_ylabel('accuracy')
    ax.set_xlabel('epoch')
    ax.legend(['Regular, train', 'Regular, val', 'Random, train', 'Random, val'])

    ax = plt.subplot(2, 2, 2)
    ax.plot(history_regu.history['loss'], 'b')
    ax.plot(history_regu.history['val_loss'], 'c')
    ax.plot(history_rand.history['loss'], 'r')
    ax.plot(history_rand.history['val_loss'], 'y')
    ax.set_title('Level of mismatch after denoising')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend(['Regular, train', 'Regular, val', 'Random, train', 'Random, val'])


def display_results(data1, result1, original, result2, data2):
    """
    Displays images from each one of the supplied arrays.
    """

    n = 4

    start_ind = 300
    indices = np.arange(start_ind, start_ind + n)  # np.random.randint(len(array1), size=n)
    images1 = data1[indices, :]
    images2 = result1[indices, :]
    images3 = original[indices, :]
    images4 = result2[indices, :]
    images5 = data2[indices, :]

    plt.figure(figsize=(12, 12))
    plt.suptitle('Examples for clean-image reconstruction')
    for i, (image1, image2, image3, image4, image5) in enumerate(zip(images1, images2, images3, images4, images5)):
        ax = plt.subplot(5, n, i + 1)
        plt.imshow(image1.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(5, n, i + 1 + n)
        plt.imshow(image2.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(5, n, i + 1 + 2 * n)
        plt.imshow(image3.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(5, n, i + 1 + 3 * n)
        plt.imshow(image4.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(5, n, i + 1 + 4 * n)
        plt.imshow(image5.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


def run_MNIST_digit_recognizer(save_dir):
    # an MNIST digit recognizer machine, published by Gregor Koehler
    # https://nextjournal.com/gkoehler/digit-recognition-with-keras

    # matplotlib.use('agg')

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # building the input vector from the 28x28 pixels
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # normalizing the data to help with the training
    X_train /= 255
    X_test /= 255

    # print the final input shape ready for training
    print("Train matrix shape", X_train.shape)
    print("Test matrix shape", X_test.shape)

    print(np.unique(y_train, return_counts=True))

    # one-hot encoding using keras' numpy-related utilities
    n_classes = 10
    print("Shape before one-hot encoding: ", y_train.shape)
    Y_train = utils.to_categorical(y_train, n_classes)
    Y_test = utils.to_categorical(y_test, n_classes)
    print("Shape after one-hot encoding: ", Y_train.shape)

    # building a linear stack of layers with the sequential model
    model = Sequential()
    model.add(layers.Dense(512, input_shape=(784,)))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(10))
    model.add(layers.Activation('softmax'))

    # compiling the sequential model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    # training the model and saving metrics in history
    history = model.fit(X_train, Y_train,
                        batch_size=128, epochs=20,
                        verbose=2,
                        validation_data=(X_test, Y_test))

    # plotting the metrics
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Digit classification accuracy on clean data')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='lower right')

    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Digit classification loss on clean data')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper right')

    plt.tight_layout()

    # saving the model
    model_name = 'keras_mnist.h5'
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)


def plot_comparison_results(accu_regu, accu_rand):
    # plot comparison between regular vs random noise
    plt.figure(figsize=(4, 4))
    plt.hist(accu_regu, bins=10, alpha=0.5, color='b')
    plt.hist(accu_rand, bins=10, alpha=0.5, color='r')
    plt.title('Histogram of digit-classification accuracy after reconstruction')
    plt.ylabel('Number of runs')
    plt.xlabel('Accuracy rate')
    plt.legend(['regular ''noise''', 'random noise'])

    # Boxplot accuracy results over multiple runs of the model
    plt.figure()
    num_runs = len(accu_rand)
    sns.set_theme(style="whitegrid")
    sns.set(rc={'figure.figsize':(12, 8)})
    noise_type = np.concatenate((['Regular']*num_runs, ['Random']*num_runs))
    accuracy = np.concatenate((accu_regu, accu_rand))
    df = pd.DataFrame(data={'Noise_type': noise_type, 'Accuracy': accuracy})
    ax = sns.boxplot(x='Noise_type', y='Accuracy', data=df).set(title='Classification accuracy after denoising, '
                                                                      'accross ' + str(num_runs) + ' runs')
    plt.show()



