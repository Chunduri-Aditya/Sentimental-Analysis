# Sentiment Analysis Project

**Author**: Chunduri Aditya  
**GitHub Username**: [Chunduri-Aditya](https://github.com/Chunduri-Aditya)  
**USC ID**: 8726443356  

## Project Overview

This project involves building a text classification model to analyze the sentiment of movie reviews. The goal is to classify the reviews as either positive or negative using different deep learning architectures. The models are developed using Keras and Python, and their performance is evaluated based on training and testing accuracy.

## Project Structure

The project is divided into the following sections:

1. **Data Exploration and Pre-processing**
2. **Word Embeddings**
3. **Modeling:**
   - Multi-Layer Perceptron (MLP)
   - Convolutional Neural Network (CNN)
   - Long Short-Term Memory (LSTM)
4. **Evaluation and Results**

## Dataset

The dataset consists of movie reviews divided into two categories:
- **Positive Reviews**: Located in the `Datasets/Project_data/pos/` directory.
- **Negative Reviews**: Located in the `Datasets/Project_data/neg/` directory.

Each review is stored in a text file. Files with numbers 0-699 are used for training, and files with numbers 700-999 are used for testing.

## Data Exploration and Pre-processing

1. **Text Cleaning**: Removal of punctuation, numbers, and stopwords.
2. **Tokenization**: Conversion of text into sequences of integers based on the frequency of words.
3. **Review Length Analysis**: Calculation of average and standard deviation of review lengths.
4. **Padding/Truncating**: Reviews are truncated or padded to a fixed length (based on the 90th percentile of review lengths).

## Word Embeddings

- **Vocabulary Size**: Limited to the top 2500 words.
- **Embedding Dimension**: Set to 32.
- **Embedding Layer**: Converts integer sequences into dense word vectors.

## Modeling

### 1. Multi-Layer Perceptron (MLP)

- **Architecture**:
  - Embedding Layer
  - Flatten Layer
  - Three Dense Layers with 50 ReLU units each
  - Dropout layers to prevent overfitting
  - Sigmoid output layer

- **Optimizer**: Adam
- **Loss Function**: Custom Binary Cross-Entropy
- **Accuracy**: 
  - Training: 49%
  - Testing: 53.5%

### 2. Convolutional Neural Network (CNN)

- **Architecture**:
  - Embedding Layer
  - Conv1D Layer with 32 filters and kernel size of 3
  - MaxPooling1D Layer
  - Flatten Layer
  - Three Dense Layers with 50 ReLU units each
  - Dropout layers to prevent overfitting
  - Sigmoid output layer

- **Optimizer**: Adam
- **Loss Function**: Custom Binary Cross-Entropy
- **Accuracy**: 
  - Training: 55.8%
  - Testing: 67.6%

### 3. Long Short-Term Memory (LSTM)

- **Architecture**:
  - Embedding Layer
  - LSTM Layer with 32 units
  - Dense Layer with 256 ReLU units
  - Dropout layers to prevent overfitting
  - Sigmoid output layer

- **Optimizer**: Adam
- **Loss Function**: Custom Binary Cross-Entropy
- **Accuracy**: 
  - Training: 80.2%
  - Testing: 79.5%

## Evaluation and Results

The project compares the performance of three different models (MLP, CNN, LSTM) in terms of training and testing accuracy. Among the models, the LSTM model showed the highest performance, indicating its effectiveness in handling sequential data for sentiment analysis.

## References

- [Keras Documentation](https://keras.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Sentiment Analysis with Python](https://towardsdatascience.com/sentiment-analysis-with-python-part-2-4f71e7bde59a)

## Usage

To run the project, ensure you have the necessary libraries installed and execute the script in a Python environment.
