# MLPvsCNN

Comparision of deep learning architectures for image classification on the CIFAR-10 dataset using TensorFlow/Keras.

## Notebook Overview: `mlpvscnn.ipynb`

[`mlpvscnn.ipynb`](mlpvscnn.ipynb) explores, tunes, and compares a **Multi-Layer Perceptron (MLP)** against a **Convolutional Neural Network (CNN)** on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset:

- **Data Pipeline**: Normalizes 32×32 color images, splits into train/validation/test sets, and handles target encoding.
- **Model Architectures & Tuning**:
  - **MLP (~3.81M parameters)**: 3 dense layers (1024, 512, 256), He normal initialization, 10% dropout, and exponential learning rate decay.
  - **CNN (~1.42M parameters)**: 3 convolutional layers (64, 128, 256 filters with 3×3 kernels), batch normalization, max pooling, and a 256-unit dense head.
- **Evaluation & Results**:
  - **Performance**: CNN achieves ~74% test accuracy compared to ~55% for the MLP.
  - **Efficiency**: The CNN converges faster and outperforms the MLP while requiring ~2.6× fewer trainable parameters.
  - **Diagnostics**: Detailed evaluation using learning curves, classification metrics (precision, recall, F1-score), confusion matrices, and qualitative sample prediction analysis.

