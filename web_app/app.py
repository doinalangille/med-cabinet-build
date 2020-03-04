from flask import Flask, json
from json import JSONEncoder
# from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def create_app():
    app = Flask(__name__)
    # cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
    data = pd.read_csv('./strains_text.csv')
    # model = joblib.load('C:/Users/doina/OneDrive/Desktop/Lambda-School/3_Data-Engineering/med-cabinet-build/strains_recomender4.joblib')

    # Model section
    def predict():
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        dtm = tfidf.fit_transform(data['Combined'])
        dtm = pd.DataFrame(dtm.todense(), columns=tfidf.get_feature_names())
        nn = NearestNeighbors(n_neighbors=4, algorithm='kd_tree')
        var = nn.fit(dtm)
        return dtm, var

    # test_string = """I would like to be happy, energetic, and to get rid of my headache. My favorite flavor is blueberry."""

    @app.route('/')
    def home():
        return "This is the Home Page"
    
    @app.route('/id/<int:strain_id>')
    # THIS IS A TEST
    def get_strains(strain_id):
        strain_index = data[data['id']==strain_id].index.values.astype(int)[0]
        results = predict()
        dtm = results[0]
        fitting = results[1]
        pred = fitting.kneighbors([dtm.iloc[strain_index].values])
        recomended = pred[1]
        return json.dumps(recomended, cls=NumpyArrayEncoder)
      
    def decode(input_str):
        return input_str.replace("%22", " ").replace("%20", " ").replace("%7B", " ").replace("%7D", " ")

    @app.errorhandler(404)
    def page_not_found(error):
        return 'This page does not exist', 404

    return app
