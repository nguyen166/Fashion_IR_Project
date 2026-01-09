"""
FashionSearch API - Main FastAPI Application
Fashion Image Retrieval with SigLIP and Relevance Feedback
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from PIL import Image
import io

from .config import settings
from .core.model import get_model
from .core.database import get_database
from .services.search import get_search_service
from .schemas import (
    TextSearchRequest,
    FeedbackRequest,
    SearchResponse,
    SearchResult,
    HealthResponse,
    ErrorResponse,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Initializes model and database on startup.
    """
    logger.info("Starting FashionSearch API...")
    
    # Initialize model (loads on first access)
    try:
        model = get_model()
        logger.info("SigLIP model initialized")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise
    
    # Initialize database connection
    try:
        db = get_database()
        logger.info(f"Connected to Milvus. Collection has {db.count()} items")
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {e}")
        raise
    
    # Initialize search service
    try:
        search_service = get_search_service()
        logger.info("Search service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize search service: {e}")
        raise
    
    logger.info("FashionSearch API is ready!")
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down FashionSearch API...")
    try:
        db.disconnect()
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title="FashionSearch API",
    description="Fashion Image Retrieval System with SigLIP embeddings and Rocchio Relevance Feedback",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware - Allow frontend from different ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # React default
        "http://localhost:5173",      # Vite default
        "http://localhost:5174",      # Vite alternative
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://localhost:8080",      # Vue default
        "*"                           # Allow all (for development)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files - Serve images from /images endpoint
app.mount("/images", StaticFiles(directory="data/fashion-dataset/images"), name="images")


# ============ Health Check ============

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Check API health status.
    Returns model and database status.
    """
    try:
        db = get_database()
        model = get_model()
        return HealthResponse(
            status="healthy",
            collection_count=db.count(),
            model_loaded=True
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            collection_count=0,
            model_loaded=False
        )


# ============ Search Endpoints ============

@app.post("/search/text", response_model=SearchResponse, tags=["Search"])
async def search_by_text(request: TextSearchRequest):
    """
    Search for fashion images using text query.
    
    **Example queries:**
    - "red dress with floral pattern"
    - "blue jeans"
    - "white sneakers"
    
    Returns similar fashion items and the query vector for feedback.
    """
    try:
        search_service = get_search_service()
        results, query_vector = search_service.search_by_text(
            query=request.query,
            top_k=request.top_k,
            category=request.category
        )
        
        return SearchResponse(
            results=[SearchResult(**r) for r in results],
            query_vector=query_vector,
            total=len(results)
        )
    except Exception as e:
        logger.error(f"Text search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/image", response_model=SearchResponse, tags=["Search"])
async def search_by_image(
    file: UploadFile = File(..., description="Image file to search with"),
    top_k: int = Query(default=10, ge=1, le=100, description="Number of results"),
    category: Optional[str] = Query(default=None, description="Filter by category")
):
    """
    Search for similar fashion images using an uploaded image.
    
    Upload a fashion image and get visually similar items.
    Returns search results and the query vector for feedback.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload an image file."
        )
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        search_service = get_search_service()
        results, query_vector = search_service.search_by_image(
            image=image,
            top_k=top_k,
            category=category
        )
        
        return SearchResponse(
            results=[SearchResult(**r) for r in results],
            query_vector=query_vector,
            total=len(results)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/feedback", response_model=SearchResponse, tags=["Search"])
async def search_with_feedback(request: FeedbackRequest):
    """
    Perform search with relevance feedback using Rocchio Algorithm.
    
    **Rocchio Formula:**
    ```
    Q_new = α * Q_old + β * (1/|D_r|) * Σ D_r - γ * (1/|D_nr|) * Σ D_nr
    ```
    
    **Parameters:**
    - `query_vector`: Original query embedding from previous search
    - `positive_ids`: IDs of images marked as relevant by user
    - `negative_ids`: IDs of images marked as not relevant
    - `alpha`: Weight for original query (default: 1.0)
    - `beta`: Weight for positive feedback (default: 0.75)
    - `gamma`: Weight for negative feedback (default: 0.25)
    
    **Workflow:**
    1. User performs initial search (text or image)
    2. User marks some results as relevant/not relevant
    3. Call this endpoint with feedback to get refined results
    """
    if not request.positive_ids and not request.negative_ids:
        raise HTTPException(
            status_code=400,
            detail="At least one positive_id or negative_id is required for feedback"
        )
    
    try:
        search_service = get_search_service()
        results, modified_vector = search_service.relevance_feedback_search(
            query_vector=request.query_vector,
            positive_ids=request.positive_ids,
            negative_ids=request.negative_ids,
            top_k=request.top_k,
            alpha=request.alpha,
            beta=request.beta,
            gamma=request.gamma
        )
        
        return SearchResponse(
            results=[SearchResult(**r) for r in results],
            query_vector=modified_vector,
            total=len(results)
        )
    except Exception as e:
        logger.error(f"Feedback search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ Info Endpoints ============

@app.get("/categories", tags=["Info"])
async def get_categories():
    """
    Get list of available fashion categories for filtering.
    Returns hardcoded list of popular categories from the dataset.
    """
    categories = [
        "Tshirts",
        "Shirts", 
        "Casual Shoes",
        "Watches",
        "Sports Shoes",
        "Kurtas",
        "Tops",
        "Handbags",
        "Heels",
        "Sunglasses",
        "Sandals",
        "Wallets",
        "Flip Flops",
        "Jeans",
        "Trousers",
        "Dresses",
        "Shorts",
        "Skirts",
        "Jackets",
        "Belts"
    ]
    return {"categories": sorted(categories)}


@app.get("/filters", tags=["Info"])
async def get_filters():
    """
    Get available filter options for metadata-based filtering.
    Returns lists of genders, colors, seasons, and usage types.
    """
    return {
        "genders": ["Men", "Women", "Boys", "Girls", "Unisex"],
        "colors": [
            "Black", "White", "Blue", "Navy Blue", "Red", "Green", "Yellow",
            "Pink", "Purple", "Grey", "Brown", "Beige", "Orange", "Maroon",
            "Khaki", "Olive", "Multi", "Cream", "Silver", "Gold"
        ],
        "seasons": ["Summer", "Winter", "Fall", "Spring"],
        "usages": ["Casual", "Formal", "Sports", "Ethnic", "Party", "Smart Casual", "Travel", "Home"]
    }


@app.get("/info/stats", tags=["Info"])
async def get_stats():
    """
    Get collection statistics.
    """
    try:
        db = get_database()
        return {
            "total_items": db.count(),
            "collection_name": settings.milvus_collection,
            "embedding_dim": settings.embedding_dim,
            "model_name": settings.model_name
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info/config", tags=["Info"])
async def get_config():
    """
    Get current configuration (non-sensitive).
    """
    return {
        "model_name": settings.model_name,
        "embedding_dim": settings.embedding_dim,
        "default_top_k": settings.default_top_k,
        "rocchio_alpha": settings.rocchio_alpha,
        "rocchio_beta": settings.rocchio_beta,
        "rocchio_gamma": settings.rocchio_gamma,
    }


# ============ Error Handlers ============

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


# ============ Entry Point ============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
