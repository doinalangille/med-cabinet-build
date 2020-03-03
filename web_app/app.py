from flask import Flask, json
from flask_cors import CORS
import pandas as pd
from sklearn.externals import joblib
from sklearn.neighbors import NearestNeighbors 

def create_app():
    app = Flask(__name__)
    cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

    @app.route('/id/<int:post_id>')
    def home(strain_id):
        data = pd.read_csv('./web_app/strains_text.csv')
        model = joblib.load('./web_app/strains_recomender4.joblib')
        strain = data[data['id'] == strain_id]
        strain = strain['Combined']

        #preds = model.kneighbors(strain)
        return strain
        
        # TO DO implement pred
        
        # 
        # df = songs100[(songs100['track_index_num'] == preds[1][0][0]) | (songs100['track_index_num'] == preds[1][0][1]) | (songs100['track_index_num'] == preds[1][0][2]) | (songs100['track_index_num'] == preds[1][0][3]) | (songs100['track_index_num'] == preds[1][0][4])]
        # dict_set = [{
        # 'track_index_num' : x[0],
        # 'track_id' : x[1],
        # 'track_name' : x[2],
        # 'artist_name' : x[3],
        # 'album_cover_url' : x[4]
        # }
        # for x in df[['track_index_num', 'track_id', 'track_name', 'artist_name', 'album_cover_url']].values]

        # json_preds = json.dumps(dict_set)

        # return json_preds
        
    
    return app