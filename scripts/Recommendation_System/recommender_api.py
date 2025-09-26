from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import random
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Global inference instance
inference = None

# Response Models
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

class RecommendationItem(BaseModel):
    item_id: str
    title: str
    main_category: str
    price: float
    average_rating: float
    rating_number: int
    mlp_score: float
    faiss_score: float

class RecommendationsResponse(BaseModel):
    user_id: str
    recommendations: List[RecommendationItem]

class SampleUsersResponse(BaseModel):
    sample_users: List[str]
    total_users: int

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global inference
    from .inference import RecommendationInference  # Direct import since we're in the same directory
    inference = RecommendationInference()
    print("Recommender System is loaded.")
    yield
    # Shutdown (cleanup if needed)
    pass

app = FastAPI(title="Recommender Demo API", lifespan=lifespan)

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok", model_loaded=inference is not None)      

@app.get("/recommend/{user_id}", response_model=RecommendationsResponse)
def get_recommendations(
    user_id: str, 
    k: int = 10, 
    faiss_N: int = 200, 
    rerank: bool = True
):
    """Get recommendations for a user"""
    if inference is None:
        raise HTTPException(status_code=503, detail="Recommender system not loaded")
    
    try:
        recommendations = inference.get_recommendations_with_details(
            user_id=user_id, 
            k=k, 
            faiss_N=faiss_N, 
            rerank=rerank
        )
        return RecommendationsResponse(user_id=user_id, recommendations=recommendations)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/users/sample", response_model=SampleUsersResponse)
def get_sample_users(count: int = 5):
    """Get a sample of available users for testing"""
    if inference is None:
        raise HTTPException(status_code=503, detail="Recommender system not loaded")
    
    available_users = list(inference.user2idx.keys())
    sample_users = random.sample(available_users, min(count, len(available_users)))
    
    return SampleUsersResponse(sample_users=sample_users, total_users=len(available_users))   
    
@app.get("/recommendation/demo", response_model=RecommendationsResponse)
def demo():
    """Get a demo recommendation for a random user"""
    if inference is None:
        raise HTTPException(status_code=503, detail="Recommender system not loaded")
    
    try:
        available_users = list(inference.user2idx.keys())
        sample_users = random.sample(available_users, min(1, len(available_users)))
        recommendations = inference.get_recommendations_with_details(
            user_id=sample_users[0], 
            k=5, 
            faiss_N=200, 
            rerank=True
        )
        return RecommendationsResponse(user_id=sample_users[0], recommendations=recommendations)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    
if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run('recommender_api:app', host='127.0.0.1', port=8000, log_level='info', timeout_keep_alive=5)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
