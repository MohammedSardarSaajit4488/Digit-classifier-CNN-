# Digit Classifier Using CNN

This project implements a **Convolutional Neural Network (CNN)** for classifying handwritten digits using the MNIST dataset. The model achieves a high accuracy of **98%** and demonstrates the power of CNNs compared to traditional machine learning methods.

## Project Overview

This project was created as part of a summer training report to showcase the implementation of a CNN-based digit classifier. The model was trained using **Python** and **TensorFlow**, and is evaluated against the popular **MNIST** dataset.

### Key Features:
- **Convolutional Layers**: For feature extraction from images.
- **Pooling Layers**: For dimensionality reduction.
- **Fully Connected Layers**: For classification.
- **Dropout Layers**: To prevent overfitting.
- **Comparison**: Performance comparison with Logistic Regression, SVM, Random Forest, and k-NN.

## Technologies Used

- **Python**
- **TensorFlow & Keras**: For building and training the CNN model.
- **NumPy**: For numerical computations and handling image data.
- **Pandas**: For data manipulation and preprocessing.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-learn**: For traditional machine learning comparisons.

## Setup and Installation

### Prerequisites

- Python 3.x installed on your machine.
- The following Python packages need to be installed:
  - `tensorflow`
  - `keras`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

You can install the required packages using `pip`:

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn

How to Run
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/digit-classifier.git
Navigate to the project directory:

bash
Copy code
cd digit-classifier
Run the Python script to train the model and evaluate the results:

bash
Copy code
python digit_classifier_cnn.py
After training, the model's accuracy will be displayed along with a confusion matrix and other metrics.

Results
Accuracy: The CNN model achieved an accuracy of 98% on the MNIST test dataset.
Performance Comparison: CNN outperformed traditional machine learning models, such as:
Logistic Regression: 91.8% accuracy
SVM: 97% accuracy
Random Forest: 95.9% accuracy
k-NN: 95.9% accuracy
Future Work
Model Optimization: Further exploration of hyperparameter tuning to improve accuracy.
Generalization: Testing on additional datasets or real-world applications.
Deployment: Implementing model compression and optimization for deployment in resource-constrained environments.
Acknowledgements
This project was guided by Anoop Garg and completed as part of a summer training program at Lovely Professional University.

License
This project is licensed under the MIT License - see the LICENSE file for details
