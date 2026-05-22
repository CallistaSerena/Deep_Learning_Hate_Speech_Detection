# Deep_Learning_Hate_Speech_Detection
## Overview
This project implements an Indonesian hate speech detection system using a hybrid deep learning architectures by combining IndoBERTweet and Bidirectional GRU. This project also implement a non transformer comparison model using TF-IDF and Bidirectional GRU.
The model is trained on Indonesian twitter text data to classify the tweets into Hate Speech (HS) and Non Hate Speech (Non-HS).

This experiment includes:
- Text preprocessing and stemming
- Data imbalance handling using synonym replacement
- Hyperparameter tuning
- Model evaluation using accuracy, precision, recall, f1-score, and confusion matrix


# Dataset
The dataset used in this project is the result of previous research and can be obtained from:
https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection


## Dataset Statistics
Total data: 13.169 data
Hate Speech (HS = 1): **5.561 data**
Non Hate Speech (HS = 0): **7.608 data**

Train-Test Split:
- Training: 80%
- Testing: 20%
- Stratified split


# Model Architectures
## 1. IndoBERTweet + Bi-GRU
Architecture Flow:
```text
Input Text
↓
Tokenizer (IndoBERTweet)
↓
IndoBERTweet Embedding
↓
Bi-GRU Classification Layer
↓
Dropout
↓
Fully Connected Layer
↓
Sigmoid Activation
↓
Binary Classification
```
## 2. TF-IDF + Bi-GRU
Architecture Flow:
```text
Input Text
↓
TF-IDF Vectorizer
↓
Linear Embedding Layer
↓
Bi-GRU Classification Layer
↓
Dropout
↓
Fully Connected Layer
↓
Sigmoid Activation
↓
Binary Classification
```


# Pre processing pipeline
- Case folding
- Non alphabetical characters removal
- Stemming (using Sastrawi)


# Oversampling
To balance the dataset, synonym replacement is used to add data for the minority class

Final balanced training data distribution:
HS: **6086 data**
Non-HS: **6086 data**


# Hyperparameter Tuning
| Hyperparameter | Value |
|---|---|
| Hidden Dimension | 128 |
| Dropout | 0.5 |
| Learning Rate | 3e-5 |
| Batch Size | 64 / 128 |
| Epochs | 10 / 20 |


# Google Colab Environment
This project was developed and tested using:
- Google Colab
- Python 3.12
- NVIDIA A100 GPU (High RAM)


# Project Structure
```text
├── re_dataset.csv
├── hate_speech_detection.ipynb
├── requirements.txt
├── README.md
```


# Evaluation Metrics
The following metrics are used:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix


# Visualization
The project also provides:
- Accuracy comparison bar chart
- Confusion matrix heatmaps
- Classification report for each model


# Author
Callista Serena Ekaputri - 2702749802
