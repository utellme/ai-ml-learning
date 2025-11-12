import wandb
import xgboost as xgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator, conlist
from typing import List
import joblib
import logging
from dotenv import load_dotenv
import os
import sys
import traceback

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class PredictionRequest(BaseModel):
    amt: float
    hour: int
    day_of_week: int
    category: str
    state: str

    @validator('hour')
    def check_hour(cls, v):
        if v < 0 or v > 23:
            raise ValueError('Hour must be between 0 and 23')
        return v

    @validator('day_of_week')
    def check_day_of_week(cls, v):
        if v < 0 or v > 6:
            raise ValueError('Day of week must be between 0 and 6')
        return v

def load_artifacts():
    try:  
        # Check if WANDB_API_KEY is set
        if not os.getenv('WANDB_API_KEY'):
            raise ValueError("WANDB_API_KEY is not set in the environment variables")

        # Check if WANDB_PROJECT is set
        if not os.getenv('WANDB_PROJECT'):
            raise ValueError("WANDB_PROJECT is not set in the environment variables")

        # Initialize WandB
        if hasattr(wandb, 'login'):
            wandb.login()
            logger.info("Successfully logged in to WandB")
        else:
            logger.warning("wandb.login() not available, continuing without login")

        # Start a new run
        if hasattr(wandb, 'init'):
            run = wandb.init(project=os.getenv('WANDB_PROJECT'), job_type="inference")
            logger.info(f"Initialized WandB run for project: {os.getenv('WANDB_PROJECT')}")
        else:
            logger.warning("wandb.init() not available, continuing without initializing a run")
            run = None

        # Load model and pipeline from Weights & Biases
        artifact = run.use_artifact('production_model:latest', type='model')
        model_dir = artifact.download()
        model_path = f"{model_dir}/production_model.pkl"

        # Load model using joblib
        model = joblib.load(model_path)
        print("Model loaded successfully")
        
        artifact = run.use_artifact('pipeline:latest', type='model')
        pipeline_path = artifact.download()
        pipeline = joblib.load(f"{pipeline_path}/pipeline.pkl")
        print("Pipeline loaded successfully")

        return model, pipeline

    except Exception as e:
        raise RuntimeError(f"Failed to load necessary artifacts: {str(e)}")

try:
    model, pipeline = load_artifacts()
    logger.info("Successfully loaded artifacts")
except Exception as e:
    logger.error(f"Failed to load artifacts: {str(e)}")


# MAIN SECTION BELOW

@app.post("/predict/")
async def predict(request: PredictionRequest):
    """
    Predict the output for the given input data using the loaded model and pipeline.
    """
    try:
        # Convert input data to DataFrame
        data = pd.DataFrame([request.dict()])
        
        # Apply preprocessing pipeline
        processed_data = pipeline.transform(data)
        
        # Make prediction
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data)[0][1]
        
        return {
            "prediction": int(prediction[0]),
            "fraud_probability": float(probability)
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Endpoint for health checks.
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
