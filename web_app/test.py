from flask import Flask, jsonify
# from flask_cors import CORS
import pandas as pd
from sklearn.externals import joblib
from sklearn.neighbors import NearestNeighbors 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer




data = pd.read_csv('C:/Users/doina/OneDrive/Desktop/Lambda-School/3_Data-Engineering/med-cabinet-build/strains_text.csv')
#model = joblib.load('C:/Users/doina/OneDrive/Desktop/Lambda-School/3_Data-Engineering/med-cabinet-build/tfidf.joblib')

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
dtm = tfidf.fit_transform(data['Combined'])
dtm = pd.DataFrame(dtm.todense(), columns=tfidf.get_feature_names())
nn = NearestNeighbors(n_neighbors=4, algorithm='kd_tree')
var = nn.fit(dtm)

print 




