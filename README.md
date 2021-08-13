# Sentiment-Analysis-of-Restaurant-Reviews

The purpose of this analysis is to build a prediction model to predict whether a review on the restaurant is positive or negative. To do so, we will work on Restaurant Review dataset, we will load it into predicitve algorithms Multinomial Naive Bayes, Bernoulli Naive Bayes and Logistic Regression. In the end, we hope to find a "best" model for predicting the review's sentiment.

Dataset: Restaurant_Reviews.tsv is a dataset from Kaggle datasets which consists of 1000 reviews on a restaurant.

To build a model to predict if review is positive or negative, following steps are performed.

Importing Dataset
Preprocessing Dataset
Vectorization
Training and Classification
Analysis Conclusion


<br/>

# Preprocessing Dataset<br/>
Each review undergoes through a preprocessing step, where all the vague information is removed.

Removing the Stopwords, numeric and speacial charecters.
Normalizing each review using the approach of stemming.

<br/>

# Vectorization
From the cleaned dataset, potential features are extracted and are converted to numerical format. The vectorization techniques are used to convert textual data to numerical format. Using vectorization, a matrix is created where each column represents a feature and each row represents an individual review.

<br/>

# Training and Classification
Further the data is splitted into training and testing set using Cross Validation technique. This data is used as input to classification algorithm.

Classification Algorithms:

Algorithms like Decision tree, Support Vector Machine, Logistic Regression, Naive Bayes were implemented and on comparing the evaluation metrics two of the algorithms gave better predictions than others.

Multinomial Naive Bayes
Bernoulli Naive Bayes
Logistic Regression

<br/>

# Analysis and Conclusion<br/>
In this study, an attempt has been made to classify sentiment analysis for restaurant reviews using machine learning techniques. Two algorithms namely Multinomial Naive Bayes and Bernoulli Naive Bayes are implemented.

Evaluation metrics used here are accuracy, precision and recall.

Using Multinomial Naive Bayes,

Accuracy of prediction is 77.67%.
Precision of prediction is 0.78.
Recall of prediction is 0.77.
Using Bernoulli Naive Bayes,

Accuracy of prediction is 77.0%.
Precision of prediction is 0.76.
Recall of prediction is 0.78.
Using Logistic Regression,

Accuracy of prediction is 76.67%.
Precision of prediction is 0.8.
Recall of prediction is 0.71.
From the above results, Multinomial Naive Bayes is slightly better method compared to Bernoulli Naive Bayes and Logistic Regression, with 77.67% accuracy which means the model built for the prediction of sentiment of the restaurant review gives 77.67% right prediction.
