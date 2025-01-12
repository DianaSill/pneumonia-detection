import os
import numpy as np
import tensorflow as tf
from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.api.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Set the paths to your dataset
TRAINING_DIR = '/Users/dianasill/Documents/Projects/PneumoniaXRayML/pneumonia_detection_model/chest_xray/train'
VALIDATION_DIR = '/Users/dianasill/Documents/Projects/PneumoniaXRayML/pneumonia_detection_model/chest_xray/val'
TEST_DIR = '/Users/dianasill/Documents/Projects/PneumoniaXRayML/pneumonia_detection_model/chest_xray/test'

# Parameters
IMG_SIZE = 150  # Resize the images to this size
BATCH_SIZE = 32
EPOCHS = 15

# Load and preprocess the dataset using image_dataset_from_directory
train_dataset = image_dataset_from_directory(
    TRAINING_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='binary',
    validation_split=0.2,
    subset="training",
    seed=123
)

val_dataset = image_dataset_from_directory(
    VALIDATION_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='binary',
    validation_split=0.2,
    subset="validation",
    seed=123
)

test_dataset = image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='binary',
    seed=123
)

# Configure datasets for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# CNN Model Architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model with a smaller learning rate
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy'])

# Early stopping and ReduceLROnPlateau to help stabilize the training
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=[early_stopping, lr_scheduler]
)

# Save the model
model.save('pneumonia_detection_model.keras')

# Plot accuracy and loss graphs
def plot_history(history):
    # Accuracy plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

plot_history(history)

# Function to predict all images in a dataset
def predict_all_images(dataset):
    true_labels = []
    pred_labels = []

    for images, labels in dataset:
        preds = model.predict(images)
        true_labels.extend(labels.numpy())
        pred_labels.extend((preds > 0.5).astype(int))

    return true_labels, pred_labels

# Function to plot confusion matrix
def plot_confusion_matrix(true_labels, pred_labels):
    # Generate confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)

    # Plot confusion matrix using Seaborn
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Get predictions for the validation set
true_labels, pred_labels = predict_all_images(val_dataset)

# Print the results for the validation set
print("Validation Set Results:")
print("Classification Report:")
print(classification_report(true_labels, pred_labels, target_names=['Normal', 'Pneumonia']))
print("\nConfusion Matrix:")
plot_confusion_matrix(true_labels, pred_labels)

# Get predictions for the test set
true_labels_test, pred_labels_test = predict_all_images(test_dataset)

# Print the results for the test set
print("Test Set Results:")
print("Classification Report:")
print(classification_report(true_labels_test, pred_labels_test, target_names=['Normal', 'Pneumonia']))
print("\nConfusion Matrix:")
plot_confusion_matrix(true_labels_test, pred_labels_test)
