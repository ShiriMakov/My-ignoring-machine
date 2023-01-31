# My ignoring machine

## Do machines care about patterns in noise? 

Real world inputs often include a certain amount of irrelevant information.
This could be entirely random, aka 'noise', however it may contain some regularities that make it structured and predictable.

I was wodering whether the fact that irrelevant information has a consistent structure affects the ability of an ML algorithm to ingore it.
Here I suggest a way to test this.

I used the MNIST dataset, classicaly used for digit-recognition tasks.
I added irrelevant information to the images and utilized the dataset to train an denoising autoencoder, namely a machine that learns to remove irrelevant information from given inputs.

The irrelevant information I added to the images were pale dots. In half of the runs, these were structured in a consistent pattern for all items in both the train and test sets, and it the other half the dots were scattered for each and every image in a random fashion.

Autoencoders were trained to clean the images from irrelevant information of either type.
Reconstruction accuracy was tested using an addition model, trained on clean MNIST images to perform digit-recognition.

Autoencoders training were run multiple times on each noise-type and the resulting accuracy rates were compared using Mann-Whitney U test.
Results show that images that were cleaned of structured 'noise' were more effectively classified later, compared to those who were cleaned of random noise.

