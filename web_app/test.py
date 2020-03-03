import pandas as pd
from sklearn.externals import joblib
from sklearn.neighbors import NearestNeighbors

def home(strain_id):
    data = pd.read_csv('C:/Users/doina/OneDrive/Desktop/Lambda-School/3_Data-Engineering/med-cabinet-build/strains_text.csv')
    model = joblib.load('C:/Users/doina/OneDrive/Desktop/Lambda-School/3_Data-Engineering/med-cabinet-build//strains_recomender4.joblib')
    strain = data[data['id'] == strain_id]
    strain = strain['Combined']
    preds = model.kneighbors(strain)
    return preds