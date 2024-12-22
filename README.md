# Robust Image Classification Using CNN

This project involves developing and evaluating a robust image classification model using Convolutional Neural Networks (CNN). The model was trained on the "Animals with Attributes 2" dataset and tested under various lighting and color manipulation conditions to assess its robustness.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Technologies Used](#technologies-used)
5. [Project Workflow](#project-workflow)
6. [Results](#results)
7. [Future Work](#future-work)
8. [Setup Instructions](#setup-instructions)

---

## Introduction

The primary objective of this project is to build a robust image classification model capable of classifying animal species. To achieve this, we utilized a CNN and evaluated its performance under multiple challenging conditions, including lighting changes and color variations.

---

## Dataset

We used the "Animals with Attributes 2" dataset, which contains:
- Images of **10 animal species**.
- Approximately **650 images per class**.

Selected classes:
- **collie**, **dolphin**, **elephant**, **fox**, **moose**, **rabbit**, **sheep**, **squirrel**, **giant panda**, **polar bear**.

### Dataset Preprocessing:
- **Resizing**: All images were resized to **128x128x3**.
- **Normalization**: Pixel values normalized to the range `[0, 1]`.
- **Data Augmentation**:
  - Rotation, Brightness Adjustments, Horizontal Flip, and Gaussian Noise.

---

## Model Architecture

### Final CNN Model:
1. **Convolutional Layers**:
   - 4 blocks with filters **32, 64, 128, and 256**.
   - Conv2D, BatchNormalization, MaxPooling2D, and Dropout.
2. **Fully Connected Layers**:
   - Dense(128) with ReLU activation and Dropout(0.5).
   - Output Layer: Dense(10) with Softmax activation.
3. **Optimizer**: Adam with an ExponentialDecay Learning Rate Scheduler.
4. **Callbacks**:
   - EarlyStopping to monitor validation loss and avoid overfitting.

---

## Technologies Used

- **Python**: Programming language.
- **TensorFlow**: Model development and training.
- **OpenCV**: Image manipulation and preprocessing.
- **Matplotlib & NumPy**: Data visualization and array processing.

---

## Project Workflow

1. **Dataset Preparation**:
   - Images resized and normalized.
   - Data augmentation applied.
2. **Model Design**:
   - Created a CNN with dropout for regularization.
   - Used an ExponentialDecay learning rate.
3. **Model Training**:
   - Trained the model with augmented data.
   - Early stopping used to optimize training.
4. **Evaluation**:
   - Performance tested on manipulated images (brightness, contrast, HSV changes).
   - Evaluated on split test sets for consistency.
5. **Enhancements**:
   - Applied Gray World color constancy for robust testing.

---

## Results

### Accuracy Metrics:

| Test Type            | Accuracy (%) |
|----------------------|--------------|
| Original             | 67.79        |
| Bright Manipulation  | 62.51        |
| Dark Manipulation    | 51.18        |
| High Contrast        | 62.51        |
| Low Contrast         | 62.15        |
| Gamma Corrected      | 66.62        |
| HSV Transformed      | 38.21        |

### Accuracy Across Split Test Sets:

| Test Set   | Accuracy (%) |
|------------|--------------|
| Test Set 1 | 65.38        |
| Test Set 2 | 69.38        |
| Test Set 3 | 68.62        |

---

## Future Work

1. **Augmentation Improvements**:
   - Use adversarial examples for robustness.
   - Add more realistic augmentations (e.g., motion blur).
2. **Architectural Enhancements**:
   - Experiment with advanced architectures like ResNet or EfficientNet.
3. **Data Analysis**:
   - Investigate specific misclassified images to identify weaknesses.

---

## Setup Instructions

### 1. Clone the Repository:
```bash
$ git clone https://github.com/your-repo/image-classification-cnn.git
$ cd image-classification-cnn
```

### 2. Install Dependencies:
```bash
$ pip install -r requirements.txt
```

### 3. Run the Project:
- Train the model:
```bash
$ python train.py
```
- Test the model:
```bash
$ python test.py
```

### 4. Visualize Results:
- Generate plots:
```bash
$ python visualize.py
```

---

## Contact
If you have any questions or suggestions, feel free to open an issue or contact the project maintainers.