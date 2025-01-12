# This script:
#       handles data loading
#       resizing
#       splitting the data

import os                                                            # Provides a way to interact with the operating system (e.g., file paths)
import numpy as np                                                   # Numerical computing library (used for handling arrays and matrices)
import tensorflow as tf                                              # TensorFlow library for machine learning and deep learning
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Used for image preprocessing and augmentation

# Paths to your dataset directories (train and test data folders)
train_dir = 'chest_xray/train'
test_dir = 'chest_xray/test'

# Image parameters for resizing and batching
img_height, img_width = 150, 150   # Set the target size for image resizing
batch_size = 32                    # Define the batch size (number of images to be processed at once)

# Set up ImageDataGenerators for data augmentation (applies random transformations to the images to improve model generalization)
train_datagen = ImageDataGenerator(
    rescale=1./255,             # Rescale image pixel values to be between 0 and 1 (normalize the images)
    rotation_range=30,          # Randomly rotate images by up to 30 degrees
    width_shift_range=0.2,      # Randomly shift images horizontally by up to 20%
    height_shift_range=0.2,     # Randomly shift images vertically by up to 20%
    shear_range=0.2,            # Apply random shearing transformations (tilting the images)
    zoom_range=0.2,             # Randomly zoom into the images by up to 20%
    horizontal_flip=True,       # Randomly flip images horizontally
    fill_mode='nearest'         # Fill any empty pixels (after transformation) with the nearest pixel value
)

# Set up a test data generator (no augmentation, just rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale the test images

# Flow training data from the directories using the train data generator
train_data = train_datagen.flow_from_directory(
    train_dir,                                  # Path to the training data
    target_size=(img_height, img_width),        # Resize all images to the target size (150x150 pixels)
    batch_size=batch_size,                      # Process images in batches of size 32
    class_mode='binary'                         # Binary classification - only two classes: Pneumonia or Normal
)

# Flow test data from the directories using the test data generator
test_data = test_datagen.flow_from_directory(
    test_dir,                                   # Path to the test data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Function to return both the training and test data generators
def get_data():
    return train_data, test_data  # Return the prepared training and testing data generators

# This code prepares the training and testing data by applying various augmentations to the training images 
# and normalizing both the training and test datasets, all while loading them directly from directories.