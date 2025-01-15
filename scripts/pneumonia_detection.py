import tensorflow as tf
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.api.preprocessing import image_dataset_from_directory
from keras.api.applications import ResNet50
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# %pip install tensorflow==2.16 keras==3.8 matplotlib seaborn scikit-learn notebook pandas

# Set the paths to dataset
TRAINING_DIR = 'chest_xray/train'
VALIDATION_DIR = 'chest_xray/val'
TEST_DIR = 'chest_xray/test'

# Parameters
IMG_SIZE = 150  # Resize the images to this size
BATCH_SIZE = 32
EPOCHS = 25

# Load and preprocess the dataset using image_dataset_from_directory
train_dataset = image_dataset_from_directory(
    TRAINING_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='binary',
    seed=123
)

val_dataset = image_dataset_from_directory(
    VALIDATION_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='binary',
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

# Extract labels from the training dataset
train_labels = np.concatenate([y.numpy() for _, y in train_dataset])

# Ensure the labels are in a flat 1D array (if not already)
train_labels = train_labels.flatten()

# Compute class weights based on the labels
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

# Create a dictionary of class weights
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Enhanced Data Augmentation Layer
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.3),  # Increased rotation range
    tf.keras.layers.RandomWidth(0.3),  # Increased width range
    tf.keras.layers.RandomHeight(0.3),  # Increased height range
    tf.keras.layers.RandomZoom(0.3),  # Increased zoom range
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomContrast(0.3),  # Increased contrast range
])

# Transfer learning with ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Freeze the base model layers initially
base_model.trainable = False

# Model Architecture
model = Sequential([
    data_augmentation,  # Data Augmentation Layer
    base_model,  # Pretrained ResNet50
    GlobalAveragePooling2D(),  # Global Average Pooling to reduce dimensions
    Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02)),  # Increased L2 regularization
    Dropout(0.7),  # Increased dropout rate to 0.7 for more regularization
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model with an initial learning rate
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy'])

# Early stopping and ReduceLROnPlateau to help stabilize the training
early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)  # Increased patience
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)  # Adjusted patience

# Train the model
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=[early_stopping, lr_scheduler]
)

# FINE-TUNE model: Unfreeze the base model layers after initial training
base_model.trainable = True  # Unfreeze all layers of the base model

# Fine-tune the top layers of ResNet50 to avoid overfitting
for layer in base_model.layers[:143]:  # Freeze the first 143 layers
    layer.trainable = False

# Recompile the model with a very low learning rate for fine-tuning
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),  # Low learning rate for fine-tuning
              metrics=['accuracy'])

# Continue training for more epochs with fine-tuning
history_finetune = model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset,
    class_weight=class_weights_dict,
    callbacks=[early_stopping, lr_scheduler]
)

# Save the model after fine-tuning
model.save('pneumonia_detection_model_finetuned.keras')

# Plot accuracy and loss graphs with clear labeling
def plot_history(history):
    plt.figure(figsize=(14, 7))

    # Training Accuracy vs. Validation Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue', linestyle='-', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', linestyle='--', marker='x')
    plt.title('Accuracy Over Epochs', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True)

    # Training Loss vs. Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', color='blue', linestyle='-', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', linestyle='--', marker='x')
    plt.title('Loss Over Epochs', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

plot_history(history_finetune)

# Function to predict all images in a dataset
def predict_all_images(dataset):
    true_labels = []
    pred_labels = []

    for images, labels in dataset:
        preds = model.predict(images)
        true_labels.extend(labels.numpy())
        pred_labels.extend((preds > 0.5).astype(int))

    return true_labels, pred_labels


# Function to plot confusion matrix with improved readability
def plot_confusion_matrix(true_labels, pred_labels):
    # Generate confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)

    # Plot confusion matrix using Seaborn
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], 
                yticklabels=['Normal', 'Pneumonia'], cbar_kws={'label': 'Number of Predictions'}, 
                annot_kws={"size": 16}, linewidths=0.5)

    # Add titles and labels
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)

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
