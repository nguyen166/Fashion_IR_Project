"""
Milvus Database management for Fashion Image Retrieval.
Handles connection, collection creation, and CRUD operations.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)

from ..config import settings

logger = logging.getLogger(__name__)


class MilvusDatabase:
    """
    Milvus database wrapper for managing fashion image embeddings.
    """
    
    _instance = None
    _connected = False
    
    def __new__(cls):
        """Singleton pattern to ensure single database connection."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize Milvus connection and collection."""
        if MilvusDatabase._connected:
            return
        
        self.host = settings.milvus_host
        self.port = settings.milvus_port
        self.collection_name = settings.milvus_collection
        self.embedding_dim = settings.embedding_dim
        
        self._connect()
        self._init_collection()
        
        MilvusDatabase._connected = True
    
    def _connect(self):
        """Establish connection to Milvus server."""
        logger.info(f"Connecting to Milvus at {self.host}:{self.port}")
        connections.connect(
            alias="default",
            host=self.host,
            port=self.port
        )
        logger.info("Connected to Milvus successfully")
    
    def _init_collection(self):
        """Initialize or load the fashion_search collection."""
        if utility.has_collection(self.collection_name):
            logger.info(f"Loading existing collection: {self.collection_name}")
            self.collection = Collection(self.collection_name)
        else:
            logger.info(f"Creating new collection: {self.collection_name}")
            self._create_collection()
        
        # Load collection into memory for search
        self.collection.load()
        logger.info(f"Collection '{self.collection_name}' is ready")
    
    def _create_collection(self):
        """Create the fashion_search collection with schema."""
        # Define schema
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=False,
                description="Image ID from styles.csv"
            ),
            FieldSchema(
                name="image_path",
                dtype=DataType.VARCHAR,
                max_length=512,
                description="Path to the image file"
            ),
            FieldSchema(
                name="category",
                dtype=DataType.VARCHAR,
                max_length=128,
                description="Fashion category (articleType)"
            ),
            FieldSchema(
                name="gender",
                dtype=DataType.VARCHAR,
                max_length=50,
                description="Gender (Men/Women/Boys/Girls/Unisex)"
            ),
            FieldSchema(
                name="baseColour",
                dtype=DataType.VARCHAR,
                max_length=50,
                description="Base color of the item"
            ),
            FieldSchema(
                name="season",
                dtype=DataType.VARCHAR,
                max_length=50,
                description="Season (Summer/Winter/Fall/Spring)"
            ),
            FieldSchema(
                name="usage",
                dtype=DataType.VARCHAR,
                max_length=50,
                description="Usage type (Casual/Formal/Sports/etc)"
            ),
            FieldSchema(
                name="productDisplayName",
                dtype=DataType.VARCHAR,
                max_length=512,
                description="Product display name"
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.embedding_dim,
                description="SigLIP embedding vector"
            ),
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Fashion image search collection with metadata"
        )
        
        # Create collection
        self.collection = Collection(
            name=self.collection_name,
            schema=schema
        )
        
        # Create IVF_FLAT index for vector search with Inner Product metric
        index_params = {
            "metric_type": "IP",  # Inner Product
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        logger.info(f"Collection '{self.collection_name}' created with IVF_FLAT index (IP metric)")
    
    def insert(
        self,
        image_ids: List[int],
        image_paths: List[str],
        categories: List[str],
        genders: List[str],
        base_colours: List[str],
        seasons: List[str],
        usages: List[str],
        product_names: List[str],
        embeddings: List[List[float]]
    ) -> List[int]:
        """
        Insert fashion items into the collection.
        
        Args:
            image_ids: List of image IDs from styles.csv
            image_paths: List of image file paths
            categories: List of fashion categories (articleType)
            genders: List of genders
            base_colours: List of base colours
            seasons: List of seasons
            usages: List of usage types
            product_names: List of product display names
            embeddings: List of embedding vectors
            
        Returns:
            List of inserted IDs
        """
        if not (len(image_ids) == len(image_paths) == len(categories) == 
                len(genders) == len(base_colours) == len(seasons) == 
                len(usages) == len(product_names) == len(embeddings)):
            raise ValueError("All input lists must have the same length")
        
        data = [
            image_ids,
            image_paths,
            categories,
            genders,
            base_colours,
            seasons,
            usages,
            product_names,
            embeddings
        ]
        
        result = self.collection.insert(data)
        self.collection.flush()
        
        logger.info(f"Inserted {len(image_paths)} items into collection")
        return result.primary_keys
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar fashion images.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_expr: Optional filter expression (e.g., 'category == "dress"')
            
        Returns:
            List of search results with id, image_path, category, metadata and score
        """
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 16}
        }
        
        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=["image_path", "category", "gender", "baseColour", "season", "usage", "productDisplayName"]
        )
        
        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.id,
                    "image_path": hit.entity.get("image_path"),
                    "category": hit.entity.get("category"),
                    "gender": hit.entity.get("gender"),
                    "baseColour": hit.entity.get("baseColour"),
                    "season": hit.entity.get("season"),
                    "usage": hit.entity.get("usage"),
                    "productDisplayName": hit.entity.get("productDisplayName"),
                    "score": hit.score
                })
        
        return formatted_results
    
    def get_vectors_by_ids(self, ids: List[int]) -> Dict[int, List[float]]:
        """
        Retrieve embedding vectors by their IDs.
        
        Args:
            ids: List of item IDs
            
        Returns:
            Dictionary mapping ID to embedding vector
        """
        if not ids:
            return {}
        
        # Query to get embeddings by IDs
        expr = f"id in {ids}"
        results = self.collection.query(
            expr=expr,
            output_fields=["id", "embedding"]
        )
        
        return {item["id"]: item["embedding"] for item in results}
    
    def get_item_by_id(self, item_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a single item by ID.
        
        Args:
            item_id: The item ID
            
        Returns:
            Item data or None if not found
        """
        expr = f"id == {item_id}"
        results = self.collection.query(
            expr=expr,
            output_fields=["id", "image_path", "category", "gender", "baseColour", "season", "usage", "productDisplayName", "embedding"]
        )
        
        if results:
            return results[0]
        return None
    
    def count(self) -> int:
        """
        Get the total number of items in the collection.
        
        Returns:
            Number of items
        """
        return self.collection.num_entities
    
    def delete_collection(self):
        """Delete the entire collection."""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' deleted")
    
    def disconnect(self):
        """Disconnect from Milvus server."""
        connections.disconnect("default")
        MilvusDatabase._connected = False
        logger.info("Disconnected from Milvus")


# Global database instance
_db_instance: MilvusDatabase = None


def get_database() -> MilvusDatabase:
    """
    Get the global database instance.
    Creates the instance if it doesn't exist.
    
    Returns:
        MilvusDatabase instance
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = MilvusDatabase()
    return _db_instance


def reset_database():
    """
    Reset the global database instance.
    Call this after deleting a collection to ensure a fresh instance is created.
    """
    global _db_instance
    _db_instance = None
    MilvusDatabase._connected = False
    MilvusDatabase._instance = None
