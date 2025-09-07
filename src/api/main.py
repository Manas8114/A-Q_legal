"""
FastAPI backend for Legal QA System
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from loguru import logger
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import LegalQASystem

# Initialize FastAPI app
app = FastAPI(
    title="Legal QA System API",
    description="A comprehensive legal question-answering system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system instance
legal_qa_system = None

# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5

class QuestionResponse(BaseModel):
    question: str
    answer: str
    source: str
    confidence: float
    classification: Dict[str, Any]
    retrieved_documents: int
    explanation: Dict[str, Any]

class SystemStatusResponse(BaseModel):
    is_initialized: bool
    dataset_stats: Optional[Dict[str, Any]]
    cache_stats: Dict[str, Any]
    retriever_info: Dict[str, Any]
    classifier_info: Dict[str, Any]
    extractive_model_info: Dict[str, Any]
    generative_model_info: Dict[str, Any]

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    is_helpful: bool
    feedback_text: Optional[str] = None

class TrainingRequest(BaseModel):
    questions: List[str]
    answers: List[str]
    contexts: List[str]
    categories: List[str]

# Dependency to get system instance
def get_system():
    global legal_qa_system
    if legal_qa_system is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    return legal_qa_system

# Startup event
@app.on_event("startup")
async def startup_event():
    global legal_qa_system
    logger.info("Starting Legal QA System API...")
    
    try:
        # Initialize system
        legal_qa_system = LegalQASystem()
        
        # Check for trained models first (priority)
        trained_model_path = "trained_legal_qa_system"
        if os.path.exists(f"{trained_model_path}_classifier.pkl"):
            try:
                legal_qa_system.load_system(trained_model_path)
                logger.info("Loaded trained system successfully")
            except Exception as e:
                logger.error(f"Failed to load trained system: {e}")
                # Fallback to dataset initialization
                legal_qa_system.is_initialized = True
        else:
            logger.warning("No trained models found. Checking for datasets...")
            
            # Check if datasets exist
            dataset_paths = {
                'constitution': 'data/constitution_qa.json',
                'crpc': 'data/crpc_qa.json',
                'ipc': 'data/ipc_qa.json'
            }
            
            # Initialize with available datasets
            available_datasets = {}
            for name, path in dataset_paths.items():
                if os.path.exists(path):
                    available_datasets[name] = path
            
            if available_datasets:
                legal_qa_system.initialize_system(available_datasets)
                logger.info("System initialized with datasets successfully")
            else:
                logger.warning("No datasets found. System will be initialized without data.")
                legal_qa_system.is_initialized = True
            
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        legal_qa_system = None

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Legal QA System API",
        "version": "1.0.0",
        "status": "running"
    }

# Health check
@app.get("/health")
async def health_check():
    global legal_qa_system
    if legal_qa_system is None:
        return {"status": "unhealthy", "message": "System not initialized"}
    
    return {
        "status": "healthy",
        "initialized": legal_qa_system.is_initialized,
        "message": "System is running"
    }

# Ask a question
@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest, system: LegalQASystem = Depends(get_system)):
    try:
        if not system.is_initialized:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        result = system.ask_question(request.question, request.top_k)
        
        return QuestionResponse(
            question=result['question'],
            answer=result['answer'],
            source=result['source'],
            confidence=result['confidence'],
            classification=result['classification'],
            retrieved_documents=result['retrieved_documents'],
            explanation=result['explanation']
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get system status
@app.get("/status", response_model=SystemStatusResponse)
async def get_status(system: LegalQASystem = Depends(get_system)):
    try:
        status = system.get_system_status()
        return SystemStatusResponse(**status)
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Submit feedback
@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest, system: LegalQASystem = Depends(get_system)):
    try:
        # Store feedback (implement feedback storage)
        logger.info(f"Received feedback: {request.is_helpful} for question: {request.question[:50]}...")
        
        # In a real implementation, you would store this feedback
        # and potentially retrain models periodically
        
        return {
            "message": "Feedback received successfully",
            "question": request.question,
            "is_helpful": request.is_helpful
        }
        
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Train models (admin endpoint)
@app.post("/train")
async def train_models(request: TrainingRequest, system: LegalQASystem = Depends(get_system)):
    try:
        if not system.is_initialized:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        # Retrain classifier
        system.classifier.train(request.questions, request.categories)
        
        # Retrain extractive model
        if len(request.questions) > 10:
            system.extractive_model.train(
                request.contexts, 
                request.questions, 
                request.answers
            )
        
        # Fine-tune generative model
        if len(request.questions) > 5:
            system.generative_model.fine_tune(
                request.contexts,
                request.questions,
                request.answers
            )
        
        return {
            "message": "Models trained successfully",
            "num_examples": len(request.questions)
        }
        
    except Exception as e:
        logger.error(f"Error training models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get cache statistics
@app.get("/cache/stats")
async def get_cache_stats(system: LegalQASystem = Depends(get_system)):
    try:
        stats = system.answer_cache.get_cache_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Clear cache
@app.delete("/cache")
async def clear_cache(system: LegalQASystem = Depends(get_system)):
    try:
        system.answer_cache.clear_cache()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Search cache
@app.get("/cache/search")
async def search_cache(query: str, limit: int = 10, system: LegalQASystem = Depends(get_system)):
    try:
        results = system.answer_cache.search_cache(query, limit)
        return {
            "query": query,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Error searching cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get model information
@app.get("/models/info")
async def get_models_info(system: LegalQASystem = Depends(get_system)):
    try:
        return {
            "classifier": {
                "is_trained": system.classifier.is_trained,
                "categories": system.classifier.categories
            },
            "extractive_model": system.extractive_model.get_model_info(),
            "generative_model": system.generative_model.get_model_info(),
            "retriever": system.retriever.get_model_info()
        }
    except Exception as e:
        logger.error(f"Error getting models info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Save system
@app.post("/save")
async def save_system(filepath: str = "saved_system", system: LegalQASystem = Depends(get_system)):
    try:
        system.save_system(filepath)
        return {"message": f"System saved to {filepath}"}
    except Exception as e:
        logger.error(f"Error saving system: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Load system
@app.post("/load")
async def load_system(filepath: str = "saved_system", system: LegalQASystem = Depends(get_system)):
    try:
        system.load_system(filepath)
        return {"message": f"System loaded from {filepath}"}
    except Exception as e:
        logger.error(f"Error loading system: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )