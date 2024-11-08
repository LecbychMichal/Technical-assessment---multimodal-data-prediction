from fastapi import FastAPI, Request
import joblib
import pandas as pd
import uvicorn
import numpy as np
app = FastAPI()

model = joblib.load(
    './checkpoints/optuna_best_model.joblib')
preprocessor = joblib.load(
    './checkpoints/preprocessing_pipeline.joblib')

@app.post("/health")
def status():
    return {"message": "OK"}


@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()
        data = {k: np.nan if v is None else v for k, v in data.items()}
        X = pd.DataFrame([data])

        if 'target' in X.columns:
            X = X.drop('target', axis=1)
        if 'description' in X.columns:
            X = X.drop('description', axis=1)

        X_train_preprocessed = preprocessor.transform(X)
        y_pred = model.predict(X_train_preprocessed)

        return {"prediction": float(y_pred[0])}
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise
