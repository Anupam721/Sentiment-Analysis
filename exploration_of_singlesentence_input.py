import joblib
joblib_file = "joblib_MultinomialNAiveBayes_Model.pkl"  
joblib_MNB_model = joblib.load(joblib_file)

##########################################################################################################
import numpy as np
import pandas as pd

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
inputStmt = "It is a friggging awesome place !!!!"
corpus = []

review = re.sub('[^a-zA-Z]', ' ', inputStmt)
review = review.lower()
review = review.split()
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
review = ' '.join(review)
corpus.append(review)

    
    
# Creating the Bag of Words model using CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer

vectorizer_file_name = "vectorizer.pkl"
loaded_vectorizer = joblib.load(vectorizer_file_name)

#X = loaded_vectorizer.transform(corpus).toarray()
#X = loaded_vectorizer.transform(corpus).toarray()


#########################################################################################################


#y_pred = joblib_MNB_model.predict(X)
#print(y_pred)
result = (joblib_MNB_model.predict(loaded_vectorizer.transform(corpus)))

if result == 0:
    print("Negative Sentiment")
else:
    print("Positive sentiment")
