import pandas as pd
from fastapi import FastAPI
#import mlflow
import joblib
import lightgbm

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}
            

@app.get("/customer/{customer_id}")
def read_id(customer_id: int):
    # Charger le dataframe
    #filepath = 'df_sample.csv'
    url = "https://media.githubusercontent.com/media/Kromette/OC-P7-Model/main/df_small.csv"
    df = pd.read_csv(url, index_col=0, encoding='utf-8')
    # Choisir le bon client
    df = df.loc[df['SK_ID_CURR'] == customer_id]
    feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    X = df[feats]
    # importer le modèle
    #model_uri = "runs:/f9cf1b9923564e9a92267bc9d35b8468/model"
    #loaded_model = mlflow.sklearn.load_model(model_uri)
    loaded_model = joblib.load("model.pkl")
    # Calculer le score
    score = loaded_model.predict_proba(X)
    response = score[0][1]
    return response