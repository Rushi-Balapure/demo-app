# -*- coding: utf-8 -*-
"""
Created on Tue May  9 18:42:46 2023

@author: rushi
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json
app=FastAPI()
class model_input(BaseModel):
    N: int
    P: int
    K: int
    temperature	: float
    humidity: float
    ph: float
    rainfall: float
    #loading the model
model=pickle.load(open('RandomForest.pkl','rb'))
@app.post('/crop_predictor')
def crop_predict(input_parameters : model_input ):
    input_data=input_parameters.json()
    input_dictionary=json.loads(input_data)
    
    n=input_dictionary['N']
    p=input_dictionary['P']
    k=input_dictionary['K']
    temp=input_dictionary['temperature']
    humidity=input_dictionary['humidity']
    ph=input_dictionary['ph']
    rainfall=input_dictionary['rainfall']

    input_list=[n,p,k,temp,humidity,ph,rainfall]
    prediction=model.predict([input_list])
    return prediction[0]
    
