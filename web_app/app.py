from flask import Flask, json
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

def create_app():
    app = Flask(__name__)
    data = pd.read_csv('./strains_text.csv')

    # Model
    def predict():
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        dtm = tfidf.fit_transform(data['Combined'])
        dtm = pd.DataFrame(dtm.todense(), columns=tfidf.get_feature_names())
        nn = NearestNeighbors(n_neighbors=4, algorithm='kd_tree')
        var = nn.fit(dtm)
        return dtm, var, tfidf

    @app.route('/')
    def home():
        return "This is the Home Page"
    
    @app.route('/<user_input>')
    def recommend(user_input):
        temp_df = pd.DataFrame([{'input' : str(user_input)}]) 
        results = predict()
        dtm = results[0]
        var = results[1]
        tfidf = results[2]
        new = tfidf.transform(temp_df.values.tolist()[0])
        pred = var.kneighbors(new.todense())
        recomended = pred[1]
        strains_info=json.loads('{}')
        for i in range(4):
            info = data.iloc[recomended[0][i]]
            info = info.to_json()
            info_name = {f'strain_{i+1}':info}
            strains_info.update(info_name)
        return json.dumps(strains_info)
        # return strains_info

    @app.errorhandler(404)
    def page_not_found(error):
        return 'This page does not exist', 404

    return app
