from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load and train model
df = pd.read_csv('canada_per_capita_income.csv')
model = LinearRegression()
model.fit(df[['year']], df[['pci']])

# FastAPI app
app = FastAPI()

# Allow frontend JS to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development only
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input format
class InputYear(BaseModel):
    year: int

@app.post("/predict")
def predict(input: InputYear):
    prediction = model.predict([[input.year]])[0][0]
    co_efficient = model.coef_[0][0]
    intercept = model.intercept_[0]

    return {"year": input.year, "predicted_pci": round(prediction, 2) , "co_efficient": round(co_efficient, 2), "intercept": round(intercept, 2)}
