import numpy as np
import matplotlib.pyplot as plt
import random


def add_irrelevant_signal(dataset, dots_num=100, starting_point=0):
    # randomize the dots number and layout
    n = random.randrange(60, 120)  # number of dots added to each image
    start_point = random.randrange(0, 10)
    dot_ind = np.arange(starting_point, len(dataset[0]), int(np.ceil(len(dataset[0]) / dots_num)))
    dots_num = len(dot_ind)  # update according to what really appears on the image

    # initiate noise components
    dataset_random_noise = dataset.copy()
    dataset_regular_noise = dataset.copy()

    # add noise
    regular_noise = np.zeros(dataset[0].shape)
    regular_noise[dot_ind] = 1
    for i in np.arange(len(dataset)):
        # place n dots randomly
        random_noise = np.zeros(dataset[0].shape)
        random_noise[random.sample(range(0, len(dataset[0])), dots_num)] = 1
        # apply dots onto the digit images
        dataset_random_noise[i] += random_noise
        dataset_regular_noise[i] += regular_noise

        # ADD THE PART THAT NORMALIZES THE VALUES TO BE BETWEEN 0-1
        # THIS WAS W TENSORFLOW FUNCTION ANF IT RETURNED tf OBJECT
        # THAT MADE AN ERROR
        # BUT I REALIZED I CAN DO TO IT x.numpy() TO MAKE IT A NUMPY ARRAY.


    # plot examples
    for n, p in enumerate(dataset_regular_noise[0:9]):
        ax = plt.subplot(3, 3, n + 1)
        plt.gray()
        ax.imshow(p.reshape(28, 28))
        ax.get_xaxis().set_ticks([])

    return dataset_regular_noise, dataset_random_noise

