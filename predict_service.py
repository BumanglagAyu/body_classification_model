#
//  predict_service.py
//  
//
//  Created by AU on 14/05/2025.
//

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import uvicorn

app = FastAPI()

# Load the trained Random Forest model
model = pickle.load(open("body_type_rf_model.pkl", "rb"))

# Load the label encoder
label_encoder = pickle.load(open("body_type_label_encoder.pkl", "rb"))

class Measurement(BaseModel):
    shoulderWidth: float
    hipWidth: float
    waist: float
    height: float

@app.post("/predict")
def predict(data: Measurement):
    features = [[
        data.shoulderWidth,
        data.hipWidth,
        data.waist,
        data.height
    ]]
    prediction = model.predict(features)
    body_type_label = label_encoder.inverse_transform(prediction)[0]
    return {"bodyType": body_type_label}
