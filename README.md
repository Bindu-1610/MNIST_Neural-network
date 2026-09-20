# MNIST Handwritten Digit Classification Using Neural Network

## Deep Learning Assignment – 2

This project implements a simple **Neural Network** using **TensorFlow/Keras** to classify handwritten digits from **0 to 9** using the **MNIST handwritten digit dataset**.

The project includes dataset exploration, image preprocessing, neural network training, model evaluation, visualization of training results, prediction on five test images, and an experiment by changing the number of neurons in the hidden layer.

---

## 1. Project Overview

Handwritten digit recognition is a common image classification problem in Machine Learning and Deep Learning.

In this project, the MNIST dataset is used to train a neural network that learns patterns from handwritten digit images and predicts the corresponding digit.

The model takes a **28 × 28 grayscale image** as input and predicts one of the ten possible classes:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

---

## 2. Objectives

The main objectives of this assignment are:

- Load and explore the MNIST dataset.
- Display sample handwritten digit images.
- Preprocess and normalize the image data.
- Build a neural network using TensorFlow/Keras.
- Compile and train the neural network.
- Evaluate the model using test accuracy.
- Visualize training and validation accuracy.
- Visualize training and validation loss.
- Test the model on five handwritten digit images.
- Compare actual and predicted labels.
- Perform an experiment by changing the number of neurons.
- Compare the original and experimental models.
- Upload the project files to GitHub.

---

## 3. Dataset

The project uses the **MNIST handwritten digit dataset**, which is available directly through TensorFlow/Keras.

| Feature | Description |
|---|---|
| Dataset | MNIST |
| Training Images | 60,000 |
| Testing Images | 10,000 |
| Image Size | 28 × 28 pixels |
| Image Type | Grayscale |
| Number of Classes | 10 |
| Classes | 0–9 |

The dataset can be loaded using:

```python
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
```

There is no need to manually download the dataset.

---

## 4. Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Jupyter Notebook

---

## 5. Libraries Used

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
```

### Library Description

**TensorFlow/Keras:** Used to create, train, and evaluate the neural network.

**NumPy:** Used for numerical operations and finding predicted labels.

**Matplotlib:** Used for displaying MNIST images and creating accuracy/loss graphs.

---

## 6. Project Workflow

```text
MNIST Dataset
      ↓
Load Dataset
      ↓
Explore Dataset
      ↓
Display Sample Images
      ↓
Normalize Pixel Values
      ↓
Build Neural Network
      ↓
Compile Model
      ↓
Train Model
      ↓
Evaluate Test Accuracy
      ↓
Visualize Accuracy and Loss
      ↓
Predict Five Test Images
      ↓
Compare Actual and Predicted Labels
      ↓
Perform Experiment
      ↓
Compare Original and Experimental Models
```

---

## 7. Data Preprocessing

The MNIST images contain pixel values ranging from **0 to 255**.

The pixel values are normalized to the range **0 to 1** using:

```python
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
```

Normalization helps the neural network train more efficiently.

---

## 8. Neural Network Architecture

The original neural network consists of the following layers:

```text
Input Image
28 × 28 pixels
      ↓
Flatten Layer
784 values
      ↓
Dense Layer
128 neurons
ReLU activation
      ↓
Output Layer
10 neurons
Softmax activation
      ↓
Predicted Digit
0–9
```

### Model Code

```python
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

### Flatten Layer

The input image has dimensions 28 × 28. The Flatten layer converts the image into 784 values.

### Dense Layer

The hidden layer contains **128 neurons** with ReLU activation.

### Output Layer

The output layer contains **10 neurons**, representing digits 0 to 9. Softmax produces the probability for each class.

---

## 9. Model Compilation

The model is compiled using the **Adam optimizer**, **Sparse Categorical Crossentropy loss**, and **accuracy metric**.

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Optimizer

Adam updates the weights of the neural network during training.

### Loss Function

Sparse Categorical Crossentropy is used because the target labels are integer values from 0 to 9.

### Evaluation Metric

Accuracy measures the percentage of correctly classified images.

---

## 10. Model Training

The original model is trained using:

```python
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_split=0.2,
    batch_size=32
)
```

### Training Parameters

| Parameter | Value |
|---|---|
| Epochs | 10 |
| Batch Size | 32 |
| Validation Split | 20% |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |

---

## 11. Model Evaluation

After training, the model is evaluated using the test dataset.

```python
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)
```

### Test Results

```text
Test Loss: __________

Test Accuracy: __________ %
```

> Fill these values using the actual results obtained from the Jupyter Notebook.

---

## 12. Visualization

The project generates training and validation accuracy graphs and training and validation loss graphs.

### Training and Validation Accuracy

```python
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
```

### Training and Validation Loss

```python
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
```

These graphs help understand the learning behavior of the neural network.

---

## 13. Prediction on Five Images

The trained model is used to predict five images from the MNIST test dataset.

```python
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
```

Example format:

```text
Image 1: Actual = 7, Predicted = 7
Image 2: Actual = 2, Predicted = 2
Image 3: Actual = 1, Predicted = 1
Image 4: Actual = 0, Predicted = 0
Image 5: Actual = 4, Predicted = 4
```

The exact predictions may vary depending on the training results.

---

## 14. Experiment

One experiment was performed by changing the number of neurons in the hidden layer.

### Original Model

```python
Dense(128, activation='relu')
```

### Experimental Model

```python
Dense(256, activation='relu')
```

The number of hidden-layer neurons was increased from **128 to 256** while keeping the other parameters unchanged.

---

## 15. Experimental Model

```python
experiment_model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])
```

The experimental model is compiled and trained using the same optimizer, loss function, epochs, batch size, and validation split as the original model.

---

## 16. Model Comparison

| Model | Hidden Neurons | Epochs | Test Accuracy |
|---|---:|---:|---:|
| Original Model | 128 | 10 | ______ % |
| Experiment Model | 256 | 10 | ______ % |

The actual values should be entered based on the output from the notebook.

---

## 17. Results

The project successfully performs handwritten digit classification using a neural network.

The following tasks were completed:

- MNIST dataset successfully loaded.
- Sample handwritten digit images displayed.
- Images normalized from 0–255 to 0–1.
- Neural network successfully trained.
- Test accuracy calculated.
- Training and validation accuracy visualized.
- Training and validation loss visualized.
- Five test images classified.
- Actual and predicted labels compared.
- An experiment was performed by increasing hidden-layer neurons from 128 to 256.
- Original and experimental model accuracy compared.

---

## 18. Advantages

- Simple and easy-to-understand neural network.
- Automatically learns patterns from images.
- Does not require manual feature extraction.
- Can classify digits from 0 to 9.
- TensorFlow/Keras provides an easy framework for implementation.
- MNIST is suitable for learning image classification.

---

## 19. Limitations

- The model is trained only on the MNIST dataset.
- Performance may vary on handwritten images that are very different from MNIST images.
- A simple Dense neural network may not perform as well as more advanced CNN architectures.
- Some visually similar digits may be incorrectly classified.

---

## 20. Future Scope

The project can be improved in the future by:

1. Using a **Convolutional Neural Network (CNN)** for better image feature extraction.
2. Adding **Dropout layers** to reduce overfitting.
3. Using data augmentation to create more training examples.
4. Testing the model on custom handwritten images.
5. Developing a web application using Streamlit.
6. Deploying the trained model as an online digit recognition application.
7. Comparing different neural network architectures.

---

## 21. Project Files

The GitHub repository can contain:

```text
MNIST-Handwritten-Digit-Classification/
│
├── MNIST_Handwritten_Digit_Classification.ipynb
├── README.md
└── report.pdf
```

### File Description

**MNIST_Handwritten_Digit_Classification.ipynb**  
Contains the complete Python implementation, model training, evaluation, graphs, predictions, and experiment.

**README.md**  
Contains the project description, methodology, architecture, results, and instructions.

**report.pdf**  
Contains the assignment report.

---

## 22. How to Run the Project

### Step 1: Install Required Libraries

Open Command Prompt or Terminal:

```bash
pip install tensorflow numpy matplotlib
```

### Step 2: Open Jupyter Notebook

```bash
jupyter notebook
```

### Step 3: Open the Notebook

Open:

```text
MNIST_Handwritten_Digit_Classification.ipynb
```

### Step 4: Run All Cells

Run the cells in order. The MNIST dataset will be downloaded automatically through TensorFlow/Keras.

---

## 23. Conclusion

This project demonstrates the implementation of a simple Deep Learning neural network for handwritten digit classification using the MNIST dataset. The model learns patterns from 28 × 28 pixel images and classifies them into ten digit classes from 0 to 9.

The project covers the complete Deep Learning workflow, including dataset loading, exploration, preprocessing, model design, compilation, training, evaluation, visualization, prediction, and experimentation.

The experiment with 256 neurons provides a comparison with the original 128-neuron model and demonstrates the effect of changing the neural network architecture.

---

## 24. Author

**Name:** Modem Himabindu  
**Course:** B.Tech – Information Technology  
**Assignment:** Deep Learning Assignment – 2

---

## 25. GitHub Repository

**GitHub Link:**

```text
https://github.com/____________________________
```

> Replace the blank GitHub link and result placeholders with your actual repository link and notebook results.
