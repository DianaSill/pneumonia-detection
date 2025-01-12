# This script defines the convolutional neural network (CNN) used for this pneumonia detection project with:
#       Convolutional layers for feature extraction.
#       MaxPooling layers to reduce dimensions.
#       Fully connected layers for decision-making.
#       A sigmoid output for binary classification (normal vs pneumonia).

import tensorflow as tf
from tensorflow.keras.models import Sequential                                       # Used to create a linear stack of layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout    # Various layers used in the model

# Function to create the model architecture
def create_model(img_height, img_width):
    model = Sequential()  # Initialize the Sequential model (a linear stack of layers)

    # Convolutional Layer 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))  
    # Add a 2D convolutional layer with 32 filters, each with a kernel size of 3x3
    # Activation function is ReLU (Rectified Linear Unit), which introduces non-linearity
    # input_shape=(img_height, img_width, 3) defines the shape of input images (height, width, 3 color channels)
    
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    # Add a MaxPooling layer with a 2x2 pool size to downsample the feature maps
    # This reduces the spatial dimensions of the image, preserving important features


    # Convolutional Layer 2
    model.add(Conv2D(64, (3, 3), activation='relu'))  
    # Add another 2D convolutional layer with 64 filters of size 3x3
    # ReLU is used as the activation function again
    
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    # Add another MaxPooling layer to downsample the feature maps


    # Convolutional Layer 3
    model.add(Conv2D(128, (3, 3), activation='relu'))  
    # Add another 2D convolutional layer with 128 filters of size 3x3
    
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    # Add a third MaxPooling layer to reduce the size of the feature maps

    model.add(Flatten())  
    # Flatten the feature maps into a 1D vector to feed it into the fully connected layers


    # Fully Connected Layer (Dense layer)
    model.add(Dense(512, activation='relu'))  
    # Add a fully connected Dense layer with 512 neurons
    # ReLU is used to introduce non-linearity to the layer

    model.add(Dropout(0.5))  
    # Add a Dropout layer with a dropout rate of 50% to prevent overfitting
    # During training, randomly set 50% of the neurons to 0 to improve generalization

    # Output Layer
    model.add(Dense(1, activation='sigmoid'))  
    # Add the output Dense layer with a single neuron (binary classification: pneumonia or normal)
    # Use the sigmoid activation function since it's a binary classification problem (output between 0 and 1)

    # Compile the model with optimizer, loss function, and evaluation metric
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Adam optimizer: an adaptive learning rate optimizer
    # Binary cross-entropy: the loss function for binary classification
    # Accuracy: the metric used to evaluate model performance during training and testing

    return model  # Return the compiled model
