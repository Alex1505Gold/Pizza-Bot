import os
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import Input, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import keras
from keras.preprocessing import image

import tensorflow_hub as hub

def plot_loss_curves(history):
    """
    Plots the curves of both loss and accuracy
    """

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(loss))

    fig, ax = plt.subplots(1, 2, figsize = (20, 5))

    # Plotting loss
    ax1 = sns.lineplot(x = epochs, y = loss, label='Training Loss', ax= ax[0])
    ax1 = sns.lineplot(x = epochs, y = val_loss, label='Validation Loss', ax= ax[0])
    ax1.set(title = 'Loss', xlabel = 'Epochs')

    # Plot accuracy
    ax2 = sns.lineplot(x = epochs, y = accuracy, label='Training Accuracy', ax= ax[1])
    ax2 = sns.lineplot(x = epochs, y = val_accuracy, label='Validation Accuracy', ax=ax[1])
    ax2.set(title = 'Accuracy', xlabel = 'Epochs')

    plt.savefig('pizza_loss.png')

if __name__ == "__main__":
    data_directory = pathlib.Path('./pizza_not_pizza')
    class_names = [item.name for item in data_directory.glob('*')][:2]
    print(class_names)

    pizza_dir = './pizza_not_pizza/pizza' 
    not_pizza_dir = './pizza_not_pizza/not_pizza'
    data_dir = './pizza_not_pizza'

    num_pizza_images = len(os.listdir(pizza_dir))
    non_pizza_images = len(os.listdir(not_pizza_dir))
    print(f'Number of Pizza images: {num_pizza_images}')
    print(f'Number of Non-Pizza images: {non_pizza_images}')

    img_size = 224
    batch_size=32

    data_gen = ImageDataGenerator(rescale=1/255.,
                                shear_range=0.2,
                                zoom_range=0.2,
                                validation_split = 0.2,
                                rotation_range=30,
                                horizontal_flip=True)

    train_data = data_gen.flow_from_directory(data_dir, 
                                            target_size = (224, 224), 
                                            batch_size = 32,
                                            subset = 'training',
                                            class_mode = 'binary')

    val_data = data_gen.flow_from_directory(data_dir, 
                                            target_size = (224, 224), 
                                            batch_size = 32,
                                            subset = 'validation',
                                            class_mode = 'binary')

    #images, labels = train_data.next()
    #print(len(images), len(labels), images[0].shape)

    tf.random.set_seed(42)

    model = Sequential([
        Input(shape = (224, 224, 3)),
        Conv2D(filters = 10, kernel_size = 2, padding = 'valid', activation = 'relu'),
        MaxPool2D(pool_size = 2),
        Conv2D(filters = 32, kernel_size = 2, padding = 'valid', activation = 'relu'),
        MaxPool2D(pool_size = 2),
        Conv2D(filters = 64, kernel_size = 2, padding = 'valid', activation = 'relu'),
        MaxPool2D(pool_size = 2),
        Flatten(),
        Dense(1, activation = 'sigmoid')
    ])

    print(model.summary())

    model.compile(loss = BinaryCrossentropy(),
                    optimizer = Adam(learning_rate = 0.0001),
                    metrics = ['accuracy'])

    history = model.fit(train_data, 
                        batch_size=32,
                        epochs=30,
                        validation_data=val_data,
                        verbose=1)

    plot_loss_curves(history)

    model.save('pizza.keras')

