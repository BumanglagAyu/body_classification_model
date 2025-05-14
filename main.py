from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load the trained model and label encoder
model = joblib.load("body_type_rf_model.pkl")
encoder = joblib.load("body_type_label_encoder.pkl")

class Measurement(BaseModel):
    gender: float  # 1.0 = Male, 2.0 = Female
    age: int
    shoulderWidth: float
    waist: float
    hips: float

@app.post("/predict")
def predict(data: Measurement):
    try:
        input_df = pd.DataFrame([{
            "Gender": data.gender,
            "Age": data.age,
            "ShoulderWidth": data.shoulderWidth,
            "Waist": data.waist,
            "Hips": data.hips
        }])
        
        prediction = model.predict(input_df)
        label = encoder.inverse_transform(prediction)[0]
        return {"bodyType": label}
    except Exception as e:
        return {"error": str(e)}