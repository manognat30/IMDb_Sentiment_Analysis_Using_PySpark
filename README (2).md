# Sentiment Analysis of IMDb Movie Reviews Using PySpark

## Overview
This project leverages **PySpark** and **Natural Language Processing (NLP)** techniques to analyze IMDb movie reviews, classifying them as positive or negative. By combining structured feature engineering, exploratory data analysis (EDA), and machine learning models, the project provides actionable insights for the entertainment industry, such as audience sentiment trends and predictors of content success.

---

## Business Context
The entertainment industry relies heavily on audience feedback for strategic decision-making. This project demonstrates how sentiment analysis can:
- **Gauge Public Reception**: Predict box office or streaming success based on sentiment.
- **Enhance Content Recommendations**: Improve personalized recommendations using sentiment patterns.
- **Optimize Marketing Strategies**: Identify popular themes for targeted promotions.
- **Guide Future Productions**: Leverage audience preferences for storyline and genre selection.

---

## Key Features
- **Scalable Data Processing**: Uses PySpark for efficient handling of 50,000 IMDb reviews.
- **Feature Engineering**:
  - Word count, exclamation mark count, positive/negative word counts.
  - Advanced text processing including tokenization, stopword removal, and lemmatization.
- **Exploratory Data Analysis (EDA)**:
  - Sentiment distributions and word usage patterns.
  - Insights into review lengths, punctuation usage, and sentiment strength.
- **Machine Learning Models**:
  - Logistic Regression, Random Forest, and Gradient-Boosted Trees for classification.
  - Performance evaluation using AUC-ROC and AUC-PR metrics.
- **Actionable Business Insights**:
  - Highlights predictors of positive and negative sentiment.
  - Demonstrates the use of sentiment data to inform industry strategies.

---

## Technical Stack
- **Data Processing**: PySpark, SparkML
- **NLP Techniques**: Tokenization, TF-IDF Vectorization, Sentiment Word Analysis
- **Machine Learning**: Logistic Regression, Random Forest, Gradient-Boosted Trees
- **Visualization**: Matplotlib, Seaborn
- **Programming Language**: Python
- **Dataset**: IMDb Large Movie Review Dataset (50,000 reviews)

---

## Workflow
1. **Data Loading**:
   - Load IMDb reviews using PySpark from structured directories.
   - Assign sentiment labels and extract ratings from filenames.
2. **Data Cleaning**:
   - Remove HTML tags and punctuation, normalize case, and handle missing values.
3. **Feature Engineering**:
   - Generate features such as word count, exclamation marks, and sentiment-specific word counts.
   - Apply TF-IDF vectorization for capturing term relevance.
4. **EDA**:
   - Analyze sentiment distribution, review lengths, and word usage patterns.
   - Identify correlations between sentiment and ratings.
5. **Model Training**:
   - Train models with a pipeline for data preparation, feature assembly, and scaling.
   - Compare models using AUC-ROC and AUC-PR metrics.
6. **Inference and Insights**:
   - Logistic Regression achieves the highest AUC-ROC (0.908).
   - Highlight business implications based on model outputs.

---

## Results
- **Best Model**: Logistic Regression with AUC-ROC score of 0.908.
- **Key Insights**:
  - Positive reviews are associated with high ratings, while negative reviews align with low ratings.
  - Exclamation marks indicate extremes in sentiment intensity.
  - Review length correlates with sentiment strength but varies for mixed feedback.

---

## Future Improvements
- Incorporate semantic embeddings (e.g., Word2Vec, BERT) for deeper context understanding.
- Explore advanced ensemble methods for more robust sentiment patterns.
- Integrate real-time data pipelines for dynamic sentiment tracking.

