# Pneumonia Detection Using Deep Learning and ResNet50

This project involves building a deep learning model to detect pneumonia from chest X-ray images. Leveraging transfer learning with ResNet50, I trained and fine-tuned a robust model to classify images into "Normal" and "Pneumonia" categories.

![xray images on loop](pneumonia-deep-learning.gif)

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Data Augmentation](#data-augmentation)
5. [Training Process](#training-process)
6. [Fine-Tuning](#fine-tuning)
7. [Evaluation](#evaluation)
8. [Results](#results)
9. [Code Files](#Code-files)
10. [Challenges Faced](#challenges-faced)
11. [Future Work](#future-work)

---

## Overview

Pneumonia is a serious respiratory illness, and early detection can significantly improve treatment outcomes. This project uses a convolutional neural network (CNN) to automate the detection of pneumonia in chest X-rays, saving time for healthcare professionals.

---

## Dataset

The dataset used is sourced from [Chest X-ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), which contains:

- **Train**: X-ray images used for model training; 624 images (390 Pneumonia, 234 Normal).
- **Validation**: Images used for tuning hyperparameters and preventing overfitting; 16 images (8 Pneumonia, 8 Normal)
- **Test**: A separate set of images to evaluate the final model's performance; 624 images (390 Pneumonia, 234 Normal)

All images were resized to **224x224 pixels**, and labels were in binary format: 
- `0`: Normal 
- `1`: Pneumonia

---

## Model Architecture

For this project, I used a combination of a custom Convolutional Neural Network (CNN) and the **ResNet50** architecture for transfer learning. Below is a breakdown of the architecture:

### Custom CNN Features:
- **Convolutional Layers**: These layers help extract spatial features from the images and identify patterns.
- **Batch Normalization**: This was added to make the training more stable and faster by normalizing inputs to each layer.
- **Dropout Layers**: Included dropout to prevent overfitting by randomly turning off some neurons during training.
- **Fully Connected Layers**: These layers are responsible for taking the extracted features and classifying them into Normal or Pneumonia.

### Transfer Learning with ResNet50:
- **Data Augmentation Layer**: To improve generalization, I used transformations like rotation, zoom, brightness, and contrast adjustments on the input images.
- **Base Model (ResNet50)**: Used a pretrained ResNet50 model (trained on ImageNet) to extract high-level features. The layers were frozen at the start to focus on transfer learning.
- **Global Average Pooling**: This reduced the size of the feature maps by averaging them, making the model more efficient.
- **Dense Layer**: Added a fully connected layer with L2 regularization to reduce overfitting.
- **Dropout Layer**: To combat overfitting even more, 70% of the units were randomly dropped in this layer.
- **Output Layer**: This layer used a sigmoid activation function for binary classification (Normal vs. Pneumonia).

### Model Compilation
We compiled the model using the following settings:
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Learning Rate**: Started with a learning rate of 0.0001 and reduced it during training to improve performance.


---

## Data Augmentation

To enhance the model's ability to generalize, advanced data augmentation techniques were used:
- **Random Rotation**: ±25% of 360 degrees
- **Random Zoom**: Up to 25% zoom in/out
- **Width/Height Shifts**: Randomly shifts the image dimensions by ±20%
- **Random Brightness/Contrast**: Enhances variability in pixel intensity.

---

## Training Process

1. **Initial Training**: The ResNet50 base model was frozen, and only the top layers (GlobalAveragePooling2D, Dense, Dropout, and Output) were trained.
   - Optimizer: Adam
   - Learning Rate: `0.0001`
   - Epochs: `30`
   - Early Stopping: Patience of `6`
   - ReduceLROnPlateau: Halves the learning rate if validation loss stagnates for `4` epochs.

2. **Class Imbalance Handling**: Used `class_weight` during training to adjust for an imbalance between Normal and Pneumonia cases.

---

## Fine-Tuning

After the initial training:

1. **Unfreezing Layers**: Unfrozen the top **100 layers** of the ResNet50 base model, allowing them to update during training.
2. **Learning Rate Schedule**: A **Cyclical Learning Rate (CLR)** was used to stabilize learning:
   - Initial Learning Rate: `1e-5`
   - Maximum Learning Rate: `1e-4`
   - Step Size: `500`
3. **Optimizer**: Adam optimizer with the cyclical learning rate schedule.

This fine-tuning step significantly improved the model's ability to learn more complex patterns in the dataset.

---

## Evaluation

1. **Confusion Matrix**:
   Visualized true positives, true negatives, false positives, and false negatives using a heatmap.
   
2. **Metrics**:
   - **Precision**
   - **Recall**
   - **F1-Score**
   - **Accuracy**

3. **Validation and Test Datasets**:
   Performance was evaluated on unseen images from both the validation and test datasets.

---

## Results

- **Validation Set (16 samples)**:
  - Accuracy: `100%`
  - Precision (Normal): `1.00`, Recall (Normal): `1.00`, F1-Score (Normal): `1.00`
  - Precision (Pneumonia): `1.00`, Recall (Pneumonia): `1.00`, F1-Score (Pneumonia): `1.00`

- **Test Set (624 samples)**:
  - Accuracy: 93%
  - Precision (Normal): `0.90`, Recall (Normal): `0.91`, F1-Score (Normal): `0.91`
  - Precision (Pneumonia): `0.95`, Recall (Pneumonia): `0.94`, F1-Score (Pneumonia): `0.94`

Confusion matrices for both validation and test sets provided deeper insights into model performance.
These images of the graphs generated from the script are in `/graphs`.

---

## Code Files

- **`pneumonia_detection_model.py`**: Python script containing the full code for training the model, including all necessary preprocessing, model architecture, training, and evaluation steps.
- **`pneumonia_detection_model.ipynb`**: Jupyter notebook containing the same code as the Python script, allowing for an easier experience where you can step through the code, visualize the results, and make modifications if desired.

Both files are functionally identical, providing flexibility in how you would like to interact with the project.


**Keras Model File**: The trained Keras model file is available upon request. Due to its large size, it's not possible to upload directly to GitHub.

---

## Challenges Faced

1. **Limited Validation Data**:
   - Validation set only had 16 images, which wasn’t enough to properly evaluate the model’s performance.
   - **Solution**: Focused on getting consistent performance on the test set and relied on metrics like precision and recall to better assess how well the model was working.

2. **Class Imbalance**:
   - The dataset had way more Pneumonia samples than Normal ones, which caused a bias during training.
   - **Solution**: Used `class_weight` to adjust for the imbalance and applied data augmentation to create a more balanced representation of both classes.

3. **Overfitting**:
   - At first, the model performed really well on the training data but struggled with validation and test data, meaning it wasn’t generalizing well.
   - **Solution**: Added regularization techniques like Dropout and L2 regularization, and also used data augmentation to help the model generalize better.

4. **Long Training Times**:
   - Training the model took a really long time, especially with data augmentation and a large dataset.
   - **Solution**: GPU acceleration was used to speed things up and optimized the architecture to make the process more efficient.

5. **Training Stability**:
   - Training was sometimes unstable, and the model would overfit after a few epochs.
   - **Solution**: Used **EarlyStopping** to stop training when the validation performance wasn’t improving and **ReduceLROnPlateau** to lower the learning rate automatically when needed.

6. **Fine-Tuning**:
   - It was tricky to figure out how many layers of ResNet50 to unfreeze for fine-tuning. If I unfreezed too many, the model started overfitting.
   - **Solution**: Experimented with unfreezing different numbers of layers and used cyclical learning rate schedules to help find the best setup.

---

## Conclusion

This project shows a successful application of deep learning in medical imaging. The model achieved an good 93% accuracy on the test set, with balanced precision and recall across both classes. This highlights the potential of AI in assisting medical professionals with pneumonia detection.

### Key Takeaways:
- The CNN effectively identified pneumonia from chest X-rays with high accuracy.
- Challenges such as class imbalance and overfitting were mitigated through preprocessing and model optimization.

---

## Future Work

There are several directions for possible future improvements and applications of this model:

- **Model Deployment**: A user-friendly interface to enable healthcare practitioners to upload and analyze chest X-rays in real-time. Additionally, converting the model to frameworks like TensorFlow Lite or ONNX will allow deployment on edge devices, making it more accessible and faster for use in clinical environments.

- **Larger and More Diverse Dataset**: Training the model on a larger, more diverse dataset would further improve its generalization and ability to detect pneumonia and other conditions across different patient populations. Can be achieved by including a wider variety of X-ray images to account for various demographic factors.

- **Explainability**: To increase trust in the model’s decisions, explainable AI techniques, such as Grad-CAM, could be integrated to visualize the parts of the X-rays the model focuses on during its decision-making process. This will help healthcare practitioners better understand and interpret the model’s predictions.

- **Multiclass Classification**: The model could be extended to classify other respiratory conditions in addition to pneumonia, such as tuberculosis or COVID-19. This would make the system more versatile and valuable in a wider range of clinical scenarios.

- **Hyperparameter Optimization**: To further refine the model’s performance, hyperparameter optimization could be automated using tools like Optuna or Hyperband, enabling more efficient and effective exploration of the model’s configuration.

---
