

from flask import Flask, render_template
from flask_mysqldb import MySQL
#from wtforms import Form, StringField, TextAreaField, PasswordField, validators
from passlib.hash import sha256_crypt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download("stopwords")
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import string
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
#import requests
#import json
app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'pooja10'
app.config['MYSQL_DB'] = 'majorproject'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
mysql = MySQL(app)
@app.route('/')
def index():
    data = pd.read_csv('C:/Users/mypc/Desktop/yelp.csv')
    def sentiment_value(para):
        analyser = SentimentIntensityAnalyzer()
        result = analyser.polarity_scores(para)
        score = result['compound']
        return round(score,1)
    sample = data.text[1]
    sample2 = data.text[31]
    print(sample)
    print('Sentiment: ')
    print(sentiment_value(sample))
    print(sample2)
    print(sentiment_value(sample2))
    all_reviews = data.text
    all_sent_values = []
    all_sentiments = []
    for i in range(0,30):
        all_sent_values.append(sentiment_value(all_reviews[i]))
        temp_data = data[0:30]
        SENTIMENT_VALUE = []
        SENTIMENT = []
    for i in range(0,30):
        sent = all_sent_values[i]
        if (sent<=1 and sent>=0.5):
          SENTIMENT.append('V.Positive')
          SENTIMENT_VALUE.append(5)
        elif (sent<0.5 and sent>0):
          SENTIMENT.append('Positive')
          SENTIMENT_VALUE.append(4)
        elif (sent==0):
          SENTIMENT.append('Neutral')
          SENTIMENT_VALUE.append(3)
        elif (sent<0 and sent>=-0.5):
          SENTIMENT.append('Negative')
          SENTIMENT_VALUE.append(2)
        else:
          SENTIMENT.append('V.Negative')
          SENTIMENT_VALUE.append(1)
    temp_data['SENTIMENT_VALUE'] = SENTIMENT_VALUE
    temp_data['SENTIMENT'] = SENTIMENT
    temp_data['text length'] = data['text'].apply(len)
    temp_data.head(100)
    filter1 = temp_data[temp_data['text'].str.contains("chicken")].count()
    filter2 = temp_data[temp_data['text'].str.contains("pizza")].count()
    filter3 = temp_data[temp_data['text'].str.contains("sandwich")].count()
    print(filter1)
    print(filter2)
    print(filter3)
    word_data = {'chicken':filter1[0], 'pizza':filter2[0], 'sandwich':filter3[0]}
    maximum = max(word_data, key= word_data.get)  # Just use 'min' instead of 'max' for minimum.
    print(maximum, word_data[maximum])
    cur = mysql.connection.cursor()
    result = cur.execute('''INSERT INTO menus (Item) VALUES (%s)''', [maximum])
    mysql.connection.commit()
    cur.close()
    return render_template('index.html',maximum = maximum)
if __name__ == '__main__':
    app.run(debug=True)