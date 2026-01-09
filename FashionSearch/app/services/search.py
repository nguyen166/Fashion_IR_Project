"""
Search service with Rocchio Algorithm for relevance feedback.
Handles image and text search with user feedback incorporation.
"""

import logging
import os
import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image

from ..config import settings
from ..core.model import SigLIPModel, get_model
from ..core.database import MilvusDatabase, get_database

logger = logging.getLogger(__name__)


# Metadata mappings for query parsing
GENDER_KEYWORDS = {
    'men': 'Men',
    'male': 'Men',
    'man': 'Men',
    'boy': 'Boys',
    'boys': 'Boys',
    'women': 'Women',
    'female': 'Women',
    'woman': 'Women',
    'girl': 'Girls',
    'girls': 'Girls',
    'unisex': 'Unisex',
}

COLOR_KEYWORDS = {
    'red': 'Red',
    'blue': 'Blue',
    'navy': 'Navy Blue',
    'navy blue': 'Navy Blue',
    'green': 'Green',
    'olive': 'Olive',
    'yellow': 'Yellow',
    'orange': 'Orange',
    'pink': 'Pink',
    'purple': 'Purple',
    'black': 'Black',
    'white': 'White',
    'grey': 'Grey',
    'gray': 'Grey',
    'brown': 'Brown',
    'beige': 'Beige',
    'cream': 'Cream',
    'maroon': 'Maroon',
    'khaki': 'Khaki',
    'teal': 'Teal',
    'turquoise': 'Turquoise Blue',
    'gold': 'Gold',
    'silver': 'Silver',
    'bronze': 'Bronze',
    'copper': 'Copper',
    'multi': 'Multi',
    'multicolor': 'Multi',
    'tan': 'Tan',
    'nude': 'Nude',
    'rust': 'Rust',
    'peach': 'Peach',
    'lavender': 'Lavender',
    'burgundy': 'Burgundy',
    'mustard': 'Mustard',
    'magenta': 'Magenta',
    'lime': 'Lime Green',
    'fluorescent': 'Fluorescent Green',
    'charcoal': 'Charcoal',
    'coral': 'Coral',
    'mauve': 'Mauve',
    'taupe': 'Taupe',
    'off white': 'Off White',
    'sea green': 'Sea Green',
    'steel': 'Steel',
    'skin': 'Skin',
    'rose': 'Rose',
    'coffee brown': 'Coffee Brown',
    'mushroom brown': 'Mushroom Brown',
    'metallic': 'Metallic',
}

SEASON_KEYWORDS = {
    'summer': 'Summer',
    'winter': 'Winter',
    'fall': 'Fall',
    'autumn': 'Fall',
    'spring': 'Spring',
}

USAGE_KEYWORDS = {
    'casual': 'Casual',
    'formal': 'Formal',
    'sports': 'Sports',
    'sport': 'Sports',
    'athletic': 'Sports',
    'ethnic': 'Ethnic',
    'party': 'Party',
    'smart casual': 'Smart Casual',
    'travel': 'Travel',
    'home': 'Home',
}


def extract_metadata_filters(query: str) -> Tuple[str, Dict[str, str]]:
    """
    Extract metadata filters from a natural language query.
    
    Args:
        query: User's search query (e.g., "a men T shirt in summer red color")
        
    Returns:
        Tuple of (cleaned_query, filters_dict)
        - cleaned_query: Query with filter keywords removed
        - filters_dict: Dictionary with detected filters {field: value}
    """
    filters = {}
    query_lower = query.lower()
    words_to_remove = []
    
    # Detect gender
    for keyword, value in GENDER_KEYWORDS.items():
        # Use word boundary matching for short words
        pattern = rf'\b{re.escape(keyword)}\b'
        if re.search(pattern, query_lower):
            filters['gender'] = value
            words_to_remove.append(keyword)
            break  # Only take first match
    
    # Detect color (check multi-word colors first)
    sorted_colors = sorted(COLOR_KEYWORDS.keys(), key=len, reverse=True)
    for keyword in sorted_colors:
        pattern = rf'\b{re.escape(keyword)}\b'
        if re.search(pattern, query_lower):
            filters['baseColour'] = COLOR_KEYWORDS[keyword]
            words_to_remove.append(keyword)
            break  # Only take first match
    
    # Detect season
    for keyword, value in SEASON_KEYWORDS.items():
        pattern = rf'\b{re.escape(keyword)}\b'
        if re.search(pattern, query_lower):
            filters['season'] = value
            words_to_remove.append(keyword)
            break
    
    # Detect usage
    sorted_usages = sorted(USAGE_KEYWORDS.keys(), key=len, reverse=True)
    for keyword in sorted_usages:
        pattern = rf'\b{re.escape(keyword)}\b'
        if re.search(pattern, query_lower):
            filters['usage'] = USAGE_KEYWORDS[keyword]
            words_to_remove.append(keyword)
            break
    
    # Clean query by removing filter keywords
    cleaned_query = query
    for word in words_to_remove:
        # Remove the word with word boundaries
        cleaned_query = re.sub(rf'\b{re.escape(word)}\b', '', cleaned_query, flags=re.IGNORECASE)
    
    # Clean up extra spaces and common filler words
    cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()
    # Remove common filler phrases that might be left
    filler_phrases = ['in color', 'color', 'in', 'for', 'a', 'the']
    for phrase in filler_phrases:
        cleaned_query = re.sub(rf'^\s*{re.escape(phrase)}\s+', '', cleaned_query, flags=re.IGNORECASE)
        cleaned_query = re.sub(rf'\s+{re.escape(phrase)}\s*$', '', cleaned_query, flags=re.IGNORECASE)
    
    cleaned_query = cleaned_query.strip()
    
    # If query is empty after cleaning, use original
    if not cleaned_query:
        cleaned_query = query
    
    logger.info(f"Query parsing: '{query}' -> cleaned='{cleaned_query}', filters={filters}")
    
    return cleaned_query, filters


def build_filter_expression(
    category: Optional[str] = None,
    filters: Optional[Dict[str, str]] = None
) -> Optional[str]:
    """
    Build a Milvus filter expression from category and metadata filters.
    
    Args:
        category: Optional category filter (from dropdown)
        filters: Optional metadata filters from query parsing
        
    Returns:
        Milvus filter expression string or None
    """
    conditions = []
    
    if category:
        conditions.append(f'category == "{category}"')
    
    if filters:
        for field, value in filters.items():
            conditions.append(f'{field} == "{value}"')
    
    if conditions:
        return ' and '.join(conditions)
    
    return None


def convert_image_path_to_url(raw_path: str) -> str:
    """
    Convert raw database image path to URL accessible by frontend.
    
    Examples:
        - './data/fashion-dataset/images/1000.jpg' -> '/images/1000.jpg'
        - 'data/fashion-dataset/images/1000.jpg' -> '/images/1000.jpg'
        - 'D:\\path\\to\\data\\fashion-dataset\\images\\1000.jpg' -> '/images/1000.jpg'
        - '/images/1000.jpg' -> '/images/1000.jpg' (already correct)
    
    Args:
        raw_path: Raw image path from database
        
    Returns:
        URL path accessible via static file serving (e.g., '/images/1000.jpg')
    """
    if not raw_path:
        return ""
    
    # If already in correct URL format, return as-is
    if raw_path.startswith("/images/"):
        return raw_path
    
    # Normalize path separators (Windows to Unix style)
    normalized_path = raw_path.replace("\\", "/")
    
    # Extract filename from the path
    # Handle various path formats:
    # - ./data/fashion-dataset/images/1000.jpg
    # - data/fashion-dataset/images/1000.jpg
    # - D:/path/to/data/fashion-dataset/images/1000.jpg
    
    # Method 1: Try to find 'images/' in path and take everything after
    if "images/" in normalized_path:
        # Get the part after 'images/'
        parts = normalized_path.split("images/")
        filename = parts[-1]  # Get the last part after 'images/'
        return f"/images/{filename}"
    
    # Method 2: Just get the filename if no 'images/' found
    filename = os.path.basename(normalized_path)
    return f"/images/{filename}"


def transform_search_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transform search results to convert image paths to URLs.
    
    Args:
        results: Raw search results from database
        
    Returns:
        Transformed results with URL-accessible image paths
    """
    transformed = []
    for result in results:
        transformed_result = result.copy()
        if "image_path" in transformed_result:
            transformed_result["image_path"] = convert_image_path_to_url(
                transformed_result["image_path"]
            )
        transformed.append(transformed_result)
    return transformed


class SearchService:
    """
    Search service for fashion image retrieval.
    Implements standard search and Rocchio Algorithm for relevance feedback.
    """
    
    def __init__(
        self,
        model: Optional[SigLIPModel] = None,
        database: Optional[MilvusDatabase] = None
    ):
        """
        Initialize search service.
        
        Args:
            model: SigLIP model instance (optional, will use global if not provided)
            database: Milvus database instance (optional, will use global if not provided)
        """
        self.model = model or get_model()
        self.database = database or get_database()
        
        # Rocchio parameters (can be overridden per request)
        self.default_alpha = settings.rocchio_alpha
        self.default_beta = settings.rocchio_beta
        self.default_gamma = settings.rocchio_gamma
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        L2 normalize a vector.
        
        Args:
            vector: Input vector
            
        Returns:
            Normalized vector
        """
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    def search_by_text(
        self,
        query: str,
        top_k: int = 10,
        category: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Search for fashion images using text query.
        Automatically extracts metadata filters from query (gender, color, season, usage).
        
        Args:
            query: Text description of desired fashion item
            top_k: Number of results to return
            category: Optional category filter (from dropdown)
            
        Returns:
            Tuple of (search results, query vector)
        """
        logger.info(f"Text search: '{query}' (top_k={top_k}, category={category})")
        
        # Extract metadata filters from query
        cleaned_query, metadata_filters = extract_metadata_filters(query)
        
        # Get text embedding using cleaned query
        query_vector = self.model.get_text_embedding(cleaned_query)
        
        # Build combined filter expression
        filter_expr = build_filter_expression(category, metadata_filters)
        
        if filter_expr:
            logger.info(f"Applied filters: {filter_expr}")
        
        # Search in Milvus
        results = self.database.search(
            query_vector=query_vector.tolist(),
            top_k=top_k,
            filter_expr=filter_expr
        )
        
        # Transform image paths to URLs
        results = transform_search_results(results)
        
        logger.info(f"Found {len(results)} results for text query")
        return results, query_vector.tolist()
    
    def search_by_image(
        self,
        image: Image.Image,
        top_k: int = 10,
        category: Optional[str] = None,
        filters: Optional[Dict[str, str]] = None
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Search for similar fashion images using an image query.
        
        Args:
            image: PIL Image object
            top_k: Number of results to return
            category: Optional category filter
            filters: Optional metadata filters (gender, baseColour, season, usage)
            
        Returns:
            Tuple of (search results, query vector)
        """
        logger.info(f"Image search (top_k={top_k}, category={category}, filters={filters})")
        
        # Get image embedding
        query_vector = self.model.get_image_embedding(image)
        
        # Build combined filter expression
        filter_expr = build_filter_expression(category, filters)
        
        # Search in Milvus
        results = self.database.search(
            query_vector=query_vector.tolist(),
            top_k=top_k,
            filter_expr=filter_expr
        )
        
        # Transform image paths to URLs
        results = transform_search_results(results)
        
        logger.info(f"Found {len(results)} results for image query")
        return results, query_vector.tolist()
    
    def search_by_vector(
        self,
        query_vector: List[float],
        top_k: int = 10,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for fashion images using a pre-computed vector.
        
        Args:
            query_vector: Pre-computed query embedding vector
            top_k: Number of results to return
            category: Optional category filter
            
        Returns:
            List of search results
        """
        # Build filter expression if category specified
        filter_expr = None
        if category:
            filter_expr = f'category == "{category}"'
        
        # Search in Milvus
        results = self.database.search(
            query_vector=query_vector,
            top_k=top_k,
            filter_expr=filter_expr
        )
        
        # Transform image paths to URLs
        return transform_search_results(results)
    
    def rocchio_feedback(
        self,
        query_vector: List[float],
        positive_ids: List[int],
        negative_ids: List[int],
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply Rocchio Algorithm for relevance feedback.
        
        Rocchio Formula:
        Q_new = α * Q_old + β * (1/|D_r|) * Σ D_r - γ * (1/|D_nr|) * Σ D_nr
        
        Where:
        - Q_old: Original query vector
        - D_r: Set of relevant (positive) document vectors
        - D_nr: Set of non-relevant (negative) document vectors
        - α, β, γ: Weights for original query, positive feedback, negative feedback
        
        Args:
            query_vector: Original query embedding vector
            positive_ids: IDs of relevant (positive) images
            negative_ids: IDs of non-relevant (negative) images
            alpha: Weight for original query (default from settings)
            beta: Weight for positive feedback (default from settings)
            gamma: Weight for negative feedback (default from settings)
            
        Returns:
            Modified query vector after applying Rocchio feedback
        """
        # Use default parameters if not specified
        alpha = alpha if alpha is not None else self.default_alpha
        beta = beta if beta is not None else self.default_beta
        gamma = gamma if gamma is not None else self.default_gamma
        
        logger.info(
            f"Applying Rocchio feedback: α={alpha}, β={beta}, γ={gamma}, "
            f"positives={len(positive_ids)}, negatives={len(negative_ids)}"
        )
        
        # Convert query vector to numpy array
        q_old = np.array(query_vector)
        
        # Initialize new query with weighted original query
        q_new = alpha * q_old
        
        # Add positive feedback component
        if positive_ids:
            positive_vectors = self.database.get_vectors_by_ids(positive_ids)
            if positive_vectors:
                positive_sum = np.zeros_like(q_old)
                for vec in positive_vectors.values():
                    positive_sum += np.array(vec)
                positive_centroid = positive_sum / len(positive_vectors)
                q_new += beta * positive_centroid
                logger.debug(f"Added positive feedback from {len(positive_vectors)} vectors")
        
        # Subtract negative feedback component
        if negative_ids:
            negative_vectors = self.database.get_vectors_by_ids(negative_ids)
            if negative_vectors:
                negative_sum = np.zeros_like(q_old)
                for vec in negative_vectors.values():
                    negative_sum += np.array(vec)
                negative_centroid = negative_sum / len(negative_vectors)
                q_new -= gamma * negative_centroid
                logger.debug(f"Subtracted negative feedback from {len(negative_vectors)} vectors")
        
        # Normalize the new query vector
        q_new = self._normalize_vector(q_new)
        
        return q_new
    
    def relevance_feedback_search(
        self,
        query_vector: List[float],
        positive_ids: List[int],
        negative_ids: List[int],
        top_k: int = 10,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        category: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Perform search with relevance feedback using Rocchio Algorithm.
        
        This method:
        1. Applies Rocchio algorithm to modify the query vector
        2. Performs a new search with the modified vector
        
        Args:
            query_vector: Original query embedding vector
            positive_ids: IDs of relevant (positive) images
            negative_ids: IDs of non-relevant (negative) images
            top_k: Number of results to return
            alpha: Weight for original query
            beta: Weight for positive feedback
            gamma: Weight for negative feedback
            category: Optional category filter
            
        Returns:
            Tuple of (search results, modified query vector)
        """
        # Apply Rocchio feedback to get new query vector
        modified_vector = self.rocchio_feedback(
            query_vector=query_vector,
            positive_ids=positive_ids,
            negative_ids=negative_ids,
            alpha=alpha,
            beta=beta,
            gamma=gamma
        )
        
        # Build filter expression if category specified
        filter_expr = None
        if category:
            filter_expr = f'category == "{category}"'
        
        # Search with modified vector
        results = self.database.search(
            query_vector=modified_vector.tolist(),
            top_k=top_k,
            filter_expr=filter_expr
        )
        
        # Transform image paths to URLs
        results = transform_search_results(results)
        
        logger.info(f"Relevance feedback search returned {len(results)} results")
        return results, modified_vector.tolist()


# Global service instance
_search_service: SearchService = None


def get_search_service() -> SearchService:
    """
    Get the global search service instance.
    Creates the instance if it doesn't exist.
    
    Returns:
        SearchService instance
    """
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
    return _search_service
