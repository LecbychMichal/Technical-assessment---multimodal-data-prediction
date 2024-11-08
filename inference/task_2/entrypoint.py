import torch
import json
import redis
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
import logging
from time import time
from inference_utils import ModelManager, RedisWrapper, REDIS_ENABLED, find_image, inverse_safe_log, ImageNotFoundException
import hashlib


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Pydantic Models
class PredictionInput(BaseModel):
    tabular_features: List[float]
    description: str
    image_data: Optional[str] = None

class PredictionResponse(BaseModel):
    prediction: float
    processing_time: float
    original_scale_prediction: float
    cache_hit: bool = False


def generate_cache_key(input_data: dict) -> str:
    """Generate a cache key from input data"""
    sorted_data = dict(sorted(input_data.items()))
    data_str = json.dumps(sorted_data, sort_keys=True)
    return hashlib.md5(data_str.encode('utf-8')).hexdigest()


# Create FastAPI app with modified startup
app = FastAPI()
model_manager = ModelManager()
redis_wrapper = RedisWrapper()

# Initialize Redis client
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

@app.on_event("startup")
async def startup_event():
    model_manager.initialize()
    await model_manager.load_models()


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_manager.model is not None,
        "device": str(model_manager.device),
        "gpu_available": torch.cuda.is_available(),
        "redis_enabled": REDIS_ENABLED
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request):
    try:
        start_time = time()
        data = await request.json()
        
        cache_key = generate_cache_key(data)
        
        cached_result = redis_wrapper.get(cache_key)
        if cached_result:
            cached_data = json.loads(cached_result)
            processing_time = time() - start_time
            logger.info(f"Cache hit for key: {cache_key}")
            return PredictionResponse(
                prediction=cached_data['prediction'],
                processing_time=processing_time,
                original_scale_prediction=cached_data['original_scale_prediction'],
                cache_hit=True
            )
        
        X = pd.DataFrame([data])
        X_preprocessed = model_manager.preprocessor.transform(X)
        tabular = torch.tensor(X_preprocessed, dtype=torch.float32)
        
        description = data['description']
        text_encoding = model_manager.tokenizer(
            description,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        try:
            image = find_image(description, "./candidateschallenge/spacecraft_images")
            image_tensor = model_manager.test_transform(image).unsqueeze(0)
        except ImageNotFoundException as e:
            raise HTTPException(status_code=404, detail=str(e))
            
        tabular = tabular.to(model_manager.device)
        text_encoding = {k: v.to(model_manager.device) for k, v in text_encoding.items()}
        image_tensor = image_tensor.to(model_manager.device)
        
        with torch.no_grad():
            output = model_manager.model(tabular, text_encoding, image_tensor)
            prediction = output.cpu().numpy().item()
            original_scale_prediction = inverse_safe_log(prediction)
        
        # Try to cache the result if Redis is connected
        if redis_wrapper.connected:
            cache_data = {
                'prediction': prediction,
                'original_scale_prediction': original_scale_prediction
            }
            redis_wrapper.setex(
                cache_key,
                3600,
                json.dumps(cache_data)
            )
            
        processing_time = time() - start_time
        
        return PredictionResponse(
            prediction=prediction,
            processing_time=processing_time,
            original_scale_prediction=original_scale_prediction,
            cache_hit=False
        )
        
    except ImageNotFoundException as e:
        logger.error(f"Image not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
