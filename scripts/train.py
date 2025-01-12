# This script:
#       loads the preprocessed data
#       builds the model
#       trains it for 10 epochs

import tensorflow as tf
from data_preprocessing import get_data    # Import the function to preprocess and load the dataset
from model import create_model             # Import the function to define the CNN model

def train_model():
    train_data, test_data = get_data()  # Load and preprocess the training and testing datasets

    # Build model
    model = create_model(150, 150)    # Create a convolutional neural network model with input dimensions (150x150)

    # Train model
    history = model.fit(              # Train the model using the training data
        train_data,                   # Training dataset
        epochs=10,                    # Number of times the model sees the entire dataset
        validation_data=test_data,    # Testing data used for validation at the end of each epoch
        batch_size=32                 # Number of samples per batch during training
    )

    # Save the trained model
    model.save('pneumonia_model.h5')  # Save the trained model to a file for future use

    return history                    # Return the training history, which contains details about accuracy, loss...

if __name__ == "__main__":
    train_model()                     # Run the train_model function when the script is executed
