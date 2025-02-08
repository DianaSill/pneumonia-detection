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


# Paths to dataset
TRAINING_DIR = 'chest_xray/train'
VALIDATION_DIR = 'chest_xray/val'
TEST_DIR = 'chest_xray/test'

# Parameters
IMG_SIZE = 224  # Resizing <
BATCH_SIZE = 32
EPOCHS = 30


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
train_labels = train_labels.flatten()  # Labels are in a FLAT 1D array

# Compute class weights based on the labels
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

# Create a dictionary of class weights
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}


# Enhanced Data Augmentation Layer using tf.keras.layers
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.25),   # Rotate up to 25% of 360
    tf.keras.layers.RandomZoom(0.25),       # Zoom randomly up to 25%
    tf.keras.layers.RandomWidth(0.2),       # Random width shift up to 20%
    tf.keras.layers.RandomHeight(0.2),      # Random height shift up to 20%
    tf.keras.layers.RandomContrast(0.3),    # Randomly adjust contrast up to 30%
    tf.keras.layers.RandomBrightness(0.2),  # Randomly adjust brightness up to 20%
])


# Transfer learning with ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # Freeze the base model layers initially

# Model Architecture
model = Sequential([
    data_augmentation,
    base_model,  # Pretrained ResNet50
    GlobalAveragePooling2D(),  # Global Average Pooling to reduce dimensions of
    Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02)),  # Dense with L2 regularization
    Dropout(0.7),  # Rate set to 0.7 for regularization
    Dense(1, activation='sigmoid')  # Output layer for binary classification - normal - pneumonia
])

# Compile the model with an initial learning rate - ADAM
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy']
)


# Help stabilize the training
early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)


# Train the model
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    class_weight=class_weights_dict,  # EXPLICITLY  use class weights
    callbacks=[early_stopping, lr_scheduler]
)


# FINE-TUNE model: Unfreeze the base model layers after initial training
base_model.trainable = True  # Unfreeze all layers of the base model

# Fine-tune more layers to avoid overfitting
for layer in base_model.layers[:100]:  # Freeze the first 100 layers
    layer.trainable = False


# ---- (checks)
# Define the custom cyclical learning rate schedule
class CyclicalLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, maximal_lr, step_size):
        self.initial_lr = initial_lr
        self.maximal_lr = maximal_lr
        self.step_size = step_size

    def __call__(self, step):
        cycle = tf.floor(1 + step / (2 * self.step_size))
        x = tf.abs(step / self.step_size - 2 * cycle + 1)
        lr = self.initial_lr + (self.maximal_lr - self.initial_lr) * tf.maximum(0., (1 - x))
        return lr
    
    def get_config(self):
        # Return a dictionary of configuration values
        return {
            "initial_lr": self.initial_lr,
            "maximal_lr": self.maximal_lr,
            "step_size": self.step_size
        }

# Desired values for initial learning rate, maximal learning rate, and step size
initial_lr = 1e-5
maximal_lr = 1e-4
step_size = 500  # Number of steps per cycle

# Instantiate the cyclical learning rate schedule
lr_schedule = CyclicalLearningRate(initial_lr, maximal_lr, step_size)

# Compile the model with the cyclical learning rate schedule
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=['accuracy']
)


# Continue training for more epochs with fine-tuning
history_finetune = model.fit(
    train_dataset,
    epochs=15,
    validation_data=val_dataset,
    class_weight=class_weights_dict,
    callbacks=[early_stopping]
)


# ---------------- Not as good - replaced with cyclical learning  - check notion on why  ----------------
# Exponential Decay Learning Rate Schedule for Fine-Tuning
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=1e-5,
#     decay_steps=1000,
#     decay_rate=0.9,
#     staircase=True
# )

# # Recompile the model with a learning rate schedule for fine-tuning
# model.compile(
#     loss='binary_crossentropy',
#     optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
#     metrics=['accuracy']
# )

# # Continue training for more epochs with fine-tuning
# history_finetune = model.fit(
#     train_dataset,
#     epochs=15,
#     validation_data=val_dataset,
#     class_weight=class_weights_dict,
#     callbacks=[early_stopping, lr_scheduler]
# )
# -------


# Save the model after fine-tuning - DON'T use h5, gives deprecated warnings - binary
model.save('pneumonia_detection_model_finetuned.keras')


# Plot accuracy and loss graphs
def plot_history(history):
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

plot_history(history_finetune)


# Function to predict all images in the dataset
def predict_all_images(dataset):
    true_labels = []
    pred_labels = []

    for images, labels in dataset:
        preds = model.predict(images)
        true_labels.extend(labels.numpy())
        pred_labels.extend((preds > 0.5).astype(int))

    return true_labels, pred_labels


# Plot confusion matrix
def plot_confusion_matrix(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


# Validate and test model
true_labels, pred_labels = predict_all_images(val_dataset)
print("Validation Set Results:")
print(classification_report(true_labels, pred_labels, target_names=['Normal', 'Pneumonia']))
plot_confusion_matrix(true_labels, pred_labels)

true_labels_test, pred_labels_test = predict_all_images(test_dataset)
print("Test Set Results:")
print(classification_report(true_labels_test, pred_labels_test, target_names=['Normal', 'Pneumonia']))
plot_confusion_matrix(true_labels_test, pred_labels_test)
