# Secure MNIST Classifier using Differential Privacy, Federated Learning, and Fully Homomorphic Encryption

This GitHub repository contains an example implementation of a secure classifier for the MNIST dataset using a combination of Differential Privacy (DP), Federated Learning (FL), and Fully Homomorphic Encryption (FHE). The project demonstrates how to train a model on distributed data while preserving privacy using state-of-the-art privacy-preserving techniques.

## Project Overview

Modern machine learning applications require both accuracy and privacy. This project aims to address these concerns by integrating three cutting-edge techniques:

1. **Differential Privacy (DP):** DP is a method to provide strong privacy guarantees by adding noise to the training process. It ensures that individual data samples cannot be easily identified in the trained model.

2. **Federated Learning (FL):** FL enables training models on distributed datasets without sharing raw data. In our case, each client trains its local model using its data and then shares model updates with the central server.

3. **Fully Homomorphic Encryption (FHE):** FHE allows computations to be performed on encrypted data, enabling privacy-preserving computation without the need to decrypt the data.

## License

This project is licensed under the MIT License - see the LICENSE file for details.