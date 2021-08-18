from flask import Flask, request, render_template
#Ml packages
import joblib
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

#import saved model file
joblib_file = "joblib_MultinomialNAiveBayes_Model.pkl"  
joblib_MNB_model = joblib.load(joblib_file)

#import saved vectorizer
vectorizer_file_name = "vectorizer.pkl"
loaded_vectorizer = joblib.load(vectorizer_file_name)


app = Flask(__name__, template_folder='template')

@app.route('/')
def my_form():
    return render_template('frontend.html')

@app.route('/', methods=['POST'])
def my_form_post():
    forminput = request.form['text']
    corpus = []
    #cleaning input-data
    review = re.sub('[^a-zA-Z]', ' ', forminput)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    result = (joblib_MNB_model.predict(loaded_vectorizer.transform(corpus)))
    sentimentResult = ""
    if result == 0:
        sentimentResult = "Negative Sentiment"
    else:
        sentimentResult = "Positive sentiment"
    return sentimentResult


if __name__ == '__main__':
    app.run(host="localhost", port=8000, debug=True)
