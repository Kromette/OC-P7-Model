from typing import Union
import pandas as pd
import numpy as np
from fastapi import FastAPI
import mlflow
import shap
import json
from pydantic import BaseModel

app = FastAPI()

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
class ShapValues(BaseModel):
    values: list
    base_values: list
    data: list


@app.get("/")
def read_root():
    model_uri = "file:///C:/Users/LN6428/Documents/P7/OC-P7/mlruns/222107729629896078/f374711e229f4188a406d396f2080269/artifacts/model"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/importance/{customer_id}")
def feature_importance(customer_id: int):
    # Charger le dataframe
    df = pd.read_csv('clean_data.csv', index_col=0)
    # Choisir le bon client
    df = df.loc[df['SK_ID_CURR'] == customer_id]
    feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    X = df[feats]
    # importer le modèle
    model_uri = "file:///C:/Users/LN6428/Documents/P7/OC-P7/mlruns/222107729629896078/f374711e229f4188a406d396f2080269/artifacts/model"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    # Feature importance
    shap.initjs()
    # Créez un explainer SHAP Tree
    explainer = shap.Explainer(loaded_model)
    # Calculez les valeurs SHAP pour les instances de test
    shap_values = explainer(X)
    #shap.waterfall_plot(shap_values[0][:, 0], max_display=20)
    values = shap_values.values
    base_values = shap_values.base_values
    data = shap_values.data

    json_dump = json.dumps({'values': values, 'base_values': base_values, 'data': data}, 
                       cls=NumpyEncoder)
    print(json_dump)

    print(values.shape)
    return {json_dump}
            

@app.get("/customer/{customer_id}")
def read_id(customer_id: int):
    # Charger le dataframe
    df = pd.read_csv('clean_data.csv', index_col=0)
    # Choisir le bon client
    df = df.loc[df['SK_ID_CURR'] == customer_id]
    feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    X = df[feats]
    # importer le modèle
    model_uri = "file:///C:/Users/LN6428/Documents/P7/OC-P7/mlruns/222107729629896078/f374711e229f4188a406d396f2080269/artifacts/model"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    # Calculer le score
    score = loaded_model.predict_proba(X)
    response = score[0][1]
    return response

@app.get("/importance/{customer_id}")
def feature_importance(customer_id: int):
    # Charger le dataframe
    df = pd.read_csv('clean_data.csv', index_col=0)
    # Choisir le bon client
    df = df.loc[df['SK_ID_CURR'] == customer_id]
    feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    X = df[feats]
    # importer le modèle
    model_uri = "file:///C:/Users/LN6428/Documents/P7/OC-P7/mlruns/222107729629896078/f374711e229f4188a406d396f2080269/artifacts/model"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    # Feature importance
    shap.initjs()
    # Créez un explainer SHAP Tree
    explainer = shap.Explainer(loaded_model)
    # Calculez les valeurs SHAP pour les instances de test
    shap_values = explainer(X)
    #shap.waterfall_plot(shap_values[0][:, 0], max_display=20)
    values = shap_values.values
    base_values = shap_values.base_values
    data = shap_values.data

    json_dump = json.dumps({'values': values, 'base_values': base_values, 'data': data}, 
                       cls=NumpyEncoder)
    print(json_dump)

    print(values.shape)
    return {json_dump}
            

@app.get("/importance/{customer_id}")
def imp(customer_id: int):
    # Charger le dataframe
    df = pd.read_csv('clean_data.csv', index_col=0)
    # Choisir le bon client
    #df = df.loc[df['SK_ID_CURR'] == customer_id]
    feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    X = df[feats]
    # importer le modèle
    model_uri = "file:///C:/Users/LN6428/Documents/P7/OC-P7/mlruns/222107729629896078/f374711e229f4188a406d396f2080269/artifacts/model"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    # Feature importance
    shap.initjs()
    # Créez un explainer SHAP Tree
    explainer = shap.Explainer(loaded_model)
    # Calculez les valeurs SHAP pour les instances de test
    shap_values = explainer(X)
    raw_values = {'values': shap_values.values.tolist(), 'base_values': shap_values.base_values.tolist(), 'data': shap_values.data.tolist()}
    structured_shap = ShapValues(**raw_values)
    print(structured_shap)
    return structured_shap