# Body Classification Model

This project implements a machine learning pipeline to classify human body types using physical measurements such as shoulder width, waist, and hips. The model is trained using a Random Forest classifier and integrated into a FastAPI backend. The classification result can be used for personalized fashion recommendations.

## Features

- Classifies body type into one of: `Hourglass`, `Triangle`, `Inverted Triangle`, `Rectangle`, or `Oval`.
- Uses shoulder width, waist, hips, age, and gender as inputs.
- Trained using Random Forest and label encoding.
- Integration-ready with Swift iOS front-end.
- Includes data preprocessing Jupyter notebook and trained model files.


## How to Use

### Backend

1. Install requirements:
    ```bash
    pip install -r requirements.txt
    ```

2. Run the FastAPI server:
    ```bash
    uvicorn run:app --reload
    ```

3. Send a POST request to `/predict` with body measurements:
    ```json
    {
      "gender": 2.0  // 1.0 = Male, 2.0 = Female",
      "age": 25,
      "shoulder": 40.5,
      "waist": 30.0,
      "hips": 38.0
    }
    ```

### iOS App (SwiftUI)

- Use `URLSession` to send user input from the UI to the API and display the predicted body type.
- Refer to `ContentView.swift` in your iOS project for integration.
