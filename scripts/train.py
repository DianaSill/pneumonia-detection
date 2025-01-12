# This script:
#       loads the preprocessed data
#       builds the model
#       trains it for 20 epochs with early stopping, class weights, and learning rate scheduler

import tensorflow as tf
from data_preprocessing import get_data    # Import the function to preprocess and load the dataset
from model import create_model             # Import the function to define the CNN model
from sklearn.utils import class_weight     # Import class_weight to handle imbalanced classes
import numpy as np                         # Import numpy for handling arrays
from tensorflow.keras.callbacks import EarlyStopping    # Prevent overfitting by using early stopping to stop training when the model's performance on the validation set stops improving
from tensorflow.keras.callbacks import ReduceLROnPlateau # Reduce learning rate when validation loss plateaus

def train_model():
    train_data, test_data = get_data()  # Load and preprocess the training and testing datasets

    # Calculate class weights to handle imbalanced classes (there are more pneumonia data over normal)
    class_weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(train_data.classes), y=train_data.classes)
    class_weights = dict(enumerate(class_weights))  # Convert class weights to dictionary format

    # Build model
    model = create_model(150, 150)    # Create a convolutional neural network model with input dimensions (150x150)

    # Early stopping callback to stop training when the validation loss doesn't improve after 'patience' epochs
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Learning rate scheduler to reduce the learning rate when the validation loss plateaus
    # 'monitor' is set to 'val_loss' to watch the validation loss
    # 'factor' controls how much to reduce the learning rate by (reduce by a factor of 0.5)
    # 'patience' specifies how many epochs to wait for an improvement before reducing the learning rate (wait for 2 epochs)
    # 'min_lr' is the lower bound for the learning rate, which won't go below this value (0.00001)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001)

    # Train model with class weights, early stopping, and learning rate scheduler
    history = model.fit(              # Train the model using the training data
        train_data,                   # Training dataset
        epochs=20,                    # Number of times the model sees the entire dataset
        validation_data=test_data,    # Testing data used for validation at the end of each epoch
        batch_size=32,                # Number of samples per batch during training
        class_weight=class_weights,   # Pass the computed class weights to handle class imbalance
        callbacks=[early_stopping, lr_scheduler]  # Add early stopping and learning rate scheduler
    )

    # Save the trained model
    model.save('pneumonia_model.h5')  # Save the trained model to a file for future use (LEGACY)
    # model.save('pneumonia_model.keras')  # This is the new recommended model format

    return history                    # Return the training history, which contains details about accuracy, loss...

if __name__ == "__main__":
    train_model()                     # Run the train_model function when the script is executed
