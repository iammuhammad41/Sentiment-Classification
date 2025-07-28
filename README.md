# Sentiment Analysis using Machine Learning Models

This repository contains a Python implementation for sentiment analysis using machine learning techniques. The primary focus of this implementation is to predict sentiment labels (Negative, Neutral, and Positive) from textual data using various machine learning models, including Logistic Regression, Naive Bayes, Support Vector Machines (SVM), and Random Forest.

The implementation involves:

* Preprocessing and cleaning textual data.
* Extracting features from text using techniques like Count Vectorization and TF-IDF.
* Training multiple machine learning models on the sentiment data.
* Evaluating the models using common metrics such as accuracy, confusion matrix, and classification report.
* Saving and loading the trained models for further use.

## Table of Contents

* [Overview](#overview)
* [Installation](#installation)
* [Dataset](#dataset)
* [Preprocessing](#preprocessing)
* [Model Training](#model-training)
* [Evaluation](#evaluation)
* [Saving and Loading Models](#saving-and-loading-models)

---

## Overview

This project is a sentiment analysis tool that:

* Loads a sentiment analysis dataset containing textual data with sentiment labels.
* Preprocesses the data by cleaning and tokenizing the text.
* Trains multiple machine learning models to predict sentiment labels from the text.
* Evaluates and compares the performance of the models using metrics like **accuracy**, **precision**, **recall**, **F1 score**, and **confusion matrix**.
* Provides functions to save and load models for making predictions on new texts.

---

## Installation

Ensure you have all the dependencies installed:

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn
```

---

## Dataset

The dataset used in this implementation comes from the **Sentiment Analysis Dataset**. It contains text data with corresponding sentiment labels (Negative, Neutral, and Positive). The data is split into training and test sets.

---

## Preprocessing

The preprocessing steps performed on the dataset include:

1. **Text Cleaning**: Removing URLs, HTML tags, special characters, and numbers.
2. **Tokenization**: Breaking the text into individual words.
3. **Removing Stopwords**: Filtering out common words (e.g., 'the', 'is', 'in') that don't add much value to sentiment analysis.
4. **Lemmatization**: Reducing words to their base or root form (e.g., 'running' becomes 'run').

### Key Preprocessing Functions:

* `clean_text`: Clean and preprocess the text data by removing URLs, punctuation, numbers, etc.
* `preprocess_text`: Apply full preprocessing pipeline (cleaning, tokenizing, stopwords removal, and lemmatization).

---

## Model Training

Various models are trained and evaluated to predict the sentiment labels:

* **Logistic Regression** (using both Count Vectorizer and TF-IDF)
* **Naive Bayes** (MultinomialNB)
* **Linear SVC** (Support Vector Classifier)
* **Random Forest Classifier**

The training and evaluation functions include:

* `train_evaluate_model`: Trains a given model (classifier) and evaluates its performance using accuracy and classification report.
* `evaluate`: Evaluates the best model using **SSIM** and **PSNR** metrics.

---

## Evaluation

The performance of the models is evaluated using:

* **Classification Report**: Includes precision, recall, F1-score for each class.
* **Confusion Matrix**: Visualized as a heatmap to show the distribution of predicted and actual sentiments.

### Evaluation Metrics:

* **Accuracy**: The percentage of correct predictions.
* **Precision**: How many selected items are relevant.
* **Recall**: How many relevant items are selected.
* **F1-Score**: The harmonic mean of precision and recall.

---

## Saving and Loading Models

Once a model is trained, it can be saved using the `save_model` function and loaded back using the `load_model` function.

* **Saving the model**: Saves the trained model to a file.
* **Loading the model**: Loads the saved model from a file for future use.

Example:

```python
save_model(best_pipeline, "sentiment_analysis_model.pkl")
loaded_model = load_model("sentiment_analysis_model.pkl")
```

---

## Example Usage

After training, you can predict sentiment for new text inputs:

```python
text = "I absolutely love this product! It's amazing!"
predicted_sentiment = predict_sentiment(text, loaded_model)
print(f"Predicted sentiment: {predicted_sentiment['sentiment_label']}")
```

For batch predictions:

```python
example_texts = [
    "I absolutely love this product! It's amazing!",
    "The service was okay, nothing special.",
    "This is the worst experience I've ever had. Terrible customer service."
]

batch_results = batch_predict(example_texts, loaded_model)
for i, result in enumerate(batch_results):
    print(f"Example {i+1}: {result['sentiment_label']}")
```

