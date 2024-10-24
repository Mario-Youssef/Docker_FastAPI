from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import joblib
import os

# Define the input model for FastAPI
class ModelInput(BaseModel):
    pclass: int
    age: float
    fare: float

# Load the trained model
model = joblib.load('models/titanic_model.pkl')

# Initialize FastAPI app
app = FastAPI()

# Define a POST route for predictions
@app.post("/predict")
async def predict(data: ModelInput):
    try:
        # Prepare input data
        input_data = np.array([data.pclass, data.age, data.fare]).reshape(1, -1)

        # Get prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        # Return response with prediction and probabilities
        return {
            "prediction": int(prediction[0]),  # 0 = Did not survive, 1 = Survived
            "probability": probability.tolist()
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

# Serve the HTML page
@app.get("/", response_class=HTMLResponse)
async def serve_html():
    with open(os.path.join(os.path.dirname(__file__), 'index.html'), 'r') as file:
        return HTMLResponse(content=file.read(), status_code=200)
