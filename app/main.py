from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


model = joblib.load("model_prophet.joblib")
last_date = joblib.load("last_date.joblib") 

app = FastAPI()

class RequestBody(BaseModel):
    year: int
    month: int

@app.post("/predict")
def predict(req: RequestBody):
    target = pd.to_datetime(f"{req.year}-{req.month:02d}-01")
    df_fut = pd.DataFrame({"ds": [target]})
    fc = model.predict(df_fut)
    return {"prediction": float(fc.loc[0, "yhat"])}
