
# Decoding Deception: A Fake News Classification Project

### Overview
This repository contains the code and resources for a Fake News Classification project. The primary goal of this project is to develop and evaluate various machine-learning models for distinguishing between real and fake news articles.


## 🛠 Skills
Python (Pandas, NumPy, Scikit-learn), NLTK, TensorFlow

## Steps Performed

[Fake news Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)

1. Exploratory Data Analysis (EDA): Analyzed the dataset to understand its structure and characteristics.
- Inspect ratio of real and fake news
- Drop extra unnamed column
- Handle Missing value
- Explore the trend in article length and news category
- Word Cloud for news titles

2. Data Processing: Preprocessed the data to prepare it for model training, including cleaning and handling missing values.

3. Modeling: Implemented various machine learning algorithms for classification:
- Naive Bayes Classifier
- Logistic Regression
- Support Vector Machine
- Random Forest Classifier
- Neural Network using Tokenization and Padding
- Long Short Term Memory (LSTM) Neural Network


## How to use

1. Clone the repository

```bash
  git clone https://github.com/abhishek-bedarkar/fake-news-classification.git
  ```
2. Install the required dependencies
```
pip install -r requirements.txt
```
3. Download the dataset and create a required directory structure
   
4. Run the Jupyter notebooks in the notebooks directory to reproduce the analysis and results.


## Accuracy Results

- Naive Bayes: 84%
- Logistic Regression: 94%
- Support Vector Machine: 94%
- Random Forest Algorithm: 95%
- Neural Network: 93%
- LSTM Neural Network: 94%


## Summary

This project showcases the application of exploratory data analysis, data preprocessing, and various machine learning algorithms for fake news classification. Multiple models were explored, ranging from traditional machine learning algorithms to complex neural network architectures. The Random Forest Classifier achieved the highest accuracy at 95%, while LSTM Neural Network and SVM demonstrated competitive performance with accuracies of 94%.

These results highlight the effectiveness of different modeling techniques in distinguishing between real and fake news articles. The project provides valuable insights into the strengths and performances of various models, offering a comprehensive exploration of approaches for fake news classification




## Support

Feel free to explore, modify, and use the code for your projects or research. If you find this work helpful, consider giving it a star!

