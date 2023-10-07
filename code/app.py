# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 13:10:08 2021

@author: HP
"""

import pandas as pd
import numpy as np
from flask import Flask,render_template,url_for,request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from lightgbm import LGBMModel,LGBMClassifier
import pickle
import joblib
import re #regular expression
import string
from nltk.corpus import stopwords
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

model= joblib.load('Drug_model.pkl')

tv=joblib.load('tv_Deployment.pkl')

app=Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])

def predict():
    
    def clean_text(text):
        '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
        text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = re.sub('\w*\d\w*', '', text)
        text = re.sub("[0-9" "]+"," ",text)
        text = re.sub('[‘’“”…]', '', text)
    
        return text
    clean = lambda x: clean_text(x)



    stop_word = pd.read_csv(r"C:\Users\user\OneDrive\Documents\TCS INTERNSHIP\code\stop.txt", header=None)
    st_word = [i for i in stop_word[0]]
    stop = stopwords.words('english')
    my_stop_words = stop.copy()
#add some more stop words
    for i in st_word:
        my_stop_words.append(i)
    def datacleanig(text):
        clean_text=text.apply(lambda x: ' '.join(word for word in x.split() if word not in my_stop_words))
        lt = WordNetLemmatizer() 
        clean_text=text.apply(lambda x: " ".join([lt.lemmatize(word) for word in x.split()]))
        return clean_text
        
    
    if request.method == 'POST':
        message = request.form['message']
        df=pd.DataFrame({'clean_text':message},index=[0])
        df.clean_text=df.clean_text.apply(clean)
        df.clean_text= datacleanig(df.clean_text)
        x=tv.transform(df.clean_text)
        dense_array = x.toarray()

# Create a DataFrame with the dense array and column names from `tv`
        count = pd.DataFrame(dense_array, columns=tv.get_feature_names_out())

        count=pd.DataFrame(x.toarray(),columns= tv.get_feature_names_out())
        my_prediction = model.predict(count)[0]
        
    return render_template('result.html',prediction = my_prediction)



#model= joblib.load('Drug_model.pkl')
#print('Result',model.predict(count))
if __name__ == '__main__':
	app.run(debug=True)