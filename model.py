import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('C:/Users/doina/OneDrive/Desktop/Lambda-School/3_Data-Engineering/med-cabinet-build/strains_text.csv')

class Predictor():
    def __init__(self):
        pass

    def transform(self, data):
        self.data = data
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        dtm = tfidf.fit_transform(self.data['Combined'])
        dtm = pd.DataFrame(dtm.todense(), columns=tfidf.get_feature_names())
        # dist_matrix  = cosine_similarity(dtm)
        # df = pd.DataFrame(dist_matrix)
        nn = NearestNeighbors(n_neighbors=4, algorithm='kd_tree')
        var = nn.fit(dtm)
        pred = var.kneighbors([dtm.iloc[0].values])
        return pred

    # def model():    
    #     # Fit on DTM
    #     dtm = tranform(data)
    #     nn = NearestNeighbors(n_neighbors=4, algorithm='kd_tree')
    #     var = nn.fit(dtm)
    #     pred = var.kneighbors([dtm.iloc[0].values])
    #     return pred






