# This script:
#       will evaluate the trained model on the test set

import tensorflow as tf
from tensorflow.keras.models import load_model                         # Import function to load a pre-trained Keras model
from data_preprocessing import get_data                                # Import the function to load and preprocess the dataset
from sklearn.metrics import classification_report, confusion_matrix    # Import functions for evaluation metrics
import numpy as np

def evaluate_model():
    _, test_data = get_data()  # Load the dataset; we're only interested in the test data here

    # Load the trained model
    model = load_model('pneumonia_model.h5')     # Load the pre-trained model saved during training
    #                 pneumonia_model.keras   need to update

    # Evaluate the model
    loss, accuracy = model.evaluate(test_data)   # Evaluate the model's performance on the test data
    print(f'Test Loss: {loss}')                  # Print the test loss (a measure of how well the model performs)
    print(f'Test Accuracy: {accuracy}')          # Print the test accuracy (percentage of correct predictions)

    # Predictions
    test_labels = test_data.classes                # Extract the true labels from the test data
    test_preds = model.predict(test_data)          # Use the model to make predictions on the test data
    test_preds = np.round(test_preds).astype(int)  # Round predictions to 0 or 1 (binary classification) and cast to integers

    # Print classification report and confusion matrix
    print(classification_report(test_labels, test_preds))  # Print precision, recall, F1-score, and support for each class
    print(confusion_matrix(test_labels, test_preds))       # Print the confusion matrix (true vs. predicted classes)

if __name__ == "__main__":
    evaluate_model()  # Run the evaluate_model function if the script is executed directly
