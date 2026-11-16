
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

saved_data = joblib.load("air_quality_model.pkl")

model = saved_data["model"]
feature_cols = saved_data["features"]

class PredictionInput(BaseModel):
    features: dict[str, float | None]

@app.get("/")
def home():
    return {"message": "Air Quality API is running"}

@app.post("/predict")
def predict(data: PredictionInput):

    input_df = pd.DataFrame(
        [data.features],
        columns=feature_cols
    )

    prediction = model.predict(input_df)[0]

    return {
        "predicted_CO": round(float(prediction), 4)
    }
