import joblib
import pandas as pd

# Load model and label encoder
model = joblib.load("body_type_rf_model.pkl")
encoder = joblib.load("body_type_label_encoder.pkl")

def predict_body_type(gender, age, shoulder, waist, hips):
    # Build input DataFrame
    input_df = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "ShoulderWidth": shoulder,
        "Waist": waist,
        "Hips": hips
    }])
    
    # Predict
    prediction = model.predict(input_df)
    label = encoder.inverse_transform(prediction)[0]
    return label

# Example test
if __name__ == "__main__":
    # Test the model with user input
    print("== Body Type Prediction Test ==")
    gender = float(input("Gender (1.0 = Male, 2.0 = Female): "))
    age = int(input("Age: "))
    shoulder = float(input("Shoulder Width: "))
    waist = float(input("Waist: "))
    hips = float(input("Hips: "))

    body_type = predict_body_type(gender, age, shoulder, waist, hips)
    print(f"\nPredicted Body Type: {body_type}")