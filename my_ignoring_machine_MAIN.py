from keras.layers import Dense
from keras import Input, Model, utils
import numpy as np
import matplotlib.pyplot as plt
import random
from AE_services import add_irrelevant_signal, read_data

# provide the number of dimensions that will decide how much the input will be compressed
encoding_dim = 15
input_img = Input(shape=(784,))
# encoded representation of input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# decoded representation of code
decoded = Dense(784, activation='sigmoid')(encoded)
# Model which take input image and shows decoded images
autoencoder = Model(inputs=input_img, outputs=decoded)
utils.plot_model(autoencoder, "my_first_model_with_shape_info.png", show_shapes=True)

# # build the encoder model and decoder model separately so that we can easily differentiate between the input and output
# # This model shows encoded images
# encoder = Model(input_img, encoded)
# # Creating a decoder model
# encoded_input = Input(shape=(encoding_dim,))
# # last layer of the autoencoder model
# decoder_layer = autoencoder.layers[-1]
# # decoder model
# decoder = Model(encoded_input, decoder_layer(encoded_input))


# compile the model with the ADAM optimizer and cross-entropy loss function fitment
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


# load the data
x_train, x_test = read_data.read_data()
# add the irrelevant signals (regular / random)
x_train_regular_noise, x_train_random_noise = add_irrelevant_signal.add_irrelevant_signal(x_train)
x_test_regular_noise, x_test_random_noise = add_irrelevant_signal.add_irrelevant_signal(x_test)

# train
autoencoder.fit(x_train_regular_noise, x_train,
                epochs=15,
                batch_size=256,
                shuffle=True,
                validation_split=0.2)

# provide new noisy input for testing the reconstruction
# encoded_img = encoder.predict(x_test_regular_noise)  # MUST UNDERSTAND WHY NOT USE autoencoder HERE!!!!!
# decoded_img = decoder.predict(encoded_img)  # AND ALSO HERE!!!!!

decoded_img = autoencoder.predict(x_test_regular_noise)

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


