"""
Pydantic schemas for API request/response models.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ============ Request Schemas ============

class TextSearchRequest(BaseModel):
    """Request schema for text-based search."""
    query: str = Field(..., description="Text query to search for fashion items")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    category: Optional[str] = Field(default=None, description="Filter by category")


class FeedbackRequest(BaseModel):
    """Request schema for relevance feedback search."""
    query_vector: List[float] = Field(..., description="Original query embedding vector")
    positive_ids: List[int] = Field(default=[], description="IDs of relevant (positive) images")
    negative_ids: List[int] = Field(default=[], description="IDs of non-relevant (negative) images")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    alpha: float = Field(default=1.0, ge=0, description="Weight for original query")
    beta: float = Field(default=0.75, ge=0, description="Weight for positive feedback")
    gamma: float = Field(default=0.25, ge=0, description="Weight for negative feedback")


# ============ Response Schemas ============

class SearchResult(BaseModel):
    """Single search result item."""
    id: int = Field(..., description="Unique identifier of the fashion item")
    image_path: str = Field(..., description="Path to the image file")
    category: str = Field(..., description="Fashion category (articleType)")
    gender: Optional[str] = Field(default=None, description="Target gender")
    baseColour: Optional[str] = Field(default=None, description="Base colour")
    season: Optional[str] = Field(default=None, description="Season")
    usage: Optional[str] = Field(default=None, description="Usage type")
    productDisplayName: Optional[str] = Field(default=None, description="Product display name")
    score: float = Field(..., description="Similarity score (higher is better)")


class SearchResponse(BaseModel):
    """Response schema for search endpoints."""
    results: List[SearchResult] = Field(default=[], description="List of search results")
    query_vector: List[float] = Field(..., description="Query embedding vector (for feedback)")
    total: int = Field(..., description="Total number of results returned")


class IngestResponse(BaseModel):
    """Response schema for data ingestion."""
    success: bool = Field(..., description="Whether ingestion was successful")
    message: str = Field(..., description="Status message")
    count: int = Field(default=0, description="Number of items ingested")


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str = Field(..., description="Service status")
    collection_count: int = Field(..., description="Number of items in collection")
    model_loaded: bool = Field(..., description="Whether embedding model is loaded")


class ErrorResponse(BaseModel):
    """Response schema for errors."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
