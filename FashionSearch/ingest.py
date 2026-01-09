#!/usr/bin/env python3
"""
Data Ingestion Script for FashionSearch

This script ingests the Fashion Product Images (Small) dataset from Kaggle
into Milvus for similarity search.

Dataset structure expected:
    data/
    ├── images/         # All images (1000.jpg, 1001.jpg, ...)
    └── styles.csv      # Metadata file with id, articleType, etc.

Usage:
    python ingest.py [--batch-size 50] [--reset] [--limit 1000]
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from PIL import Image
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from app.config import settings
from app.core.model import get_model
from app.core.database import get_database, MilvusDatabase, reset_database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data/fashion-dataset")
IMAGES_DIR = DATA_DIR / "images"
STYLES_CSV = DATA_DIR / "styles.csv"


def load_metadata() -> pd.DataFrame:
    """
    Load metadata from styles.csv.
    
    Returns:
        DataFrame with id, articleType, gender, baseColour, season, usage, productDisplayName
    """
    logger.info(f"Loading metadata from {STYLES_CSV}...")
    
    if not STYLES_CSV.exists():
        raise FileNotFoundError(f"Metadata file not found: {STYLES_CSV}")
    
    # Read CSV with error handling for bad lines
    df = pd.read_csv(
        STYLES_CSV,
        on_bad_lines='skip',  # Skip malformed rows
        usecols=['id', 'articleType', 'gender', 'baseColour', 'season', 'usage', 'productDisplayName'],
        dtype={
            'id': int, 
            'articleType': str,
            'gender': str,
            'baseColour': str,
            'season': str,
            'usage': str,
            'productDisplayName': str
        }
    )
    
    # Fill NaN values with "Unknown"
    df = df.fillna("Unknown")
    
    logger.info(f"Loaded {len(df)} metadata entries")
    logger.info(f"Unique categories: {df['articleType'].nunique()}")
    logger.info(f"Unique genders: {df['gender'].nunique()}")
    logger.info(f"Unique colours: {df['baseColour'].nunique()}")
    logger.info(f"Unique seasons: {df['season'].nunique()}")
    
    return df


def get_image_files() -> List[Path]:
    """
    Get all .jpg files in the images directory.
    
    Returns:
        List of image file paths
    """
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images directory not found: {IMAGES_DIR}")
    
    image_files = list(IMAGES_DIR.glob("*.jpg"))
    logger.info(f"Found {len(image_files)} image files")
    
    return image_files


def parse_image_id(filename: str) -> Optional[int]:
    """
    Parse image ID from filename (e.g., '1000.jpg' -> 1000).
    
    Args:
        filename: Image filename
        
    Returns:
        Image ID as integer, or None if parsing fails
    """
    try:
        # Remove extension and convert to int
        return int(Path(filename).stem)
    except ValueError:
        return None


def ingest_data(
    batch_size: int = 50,
    reset: bool = False,
    limit: Optional[int] = None
) -> int:
    """
    Main ingestion function.
    
    Args:
        batch_size: Number of images per batch insert
        reset: Whether to delete existing collection first
        limit: Maximum number of images to process (None for all)
        
    Returns:
        Number of items successfully ingested
    """
    # Step 1: Load metadata
    metadata_df = load_metadata()
    # Create a dictionary for fast lookup by ID
    metadata_dict = metadata_df.set_index('id').to_dict('index')
    
    # Step 2: Get image files
    image_files = get_image_files()
    
    # Apply limit if specified
    if limit:
        image_files = image_files[:limit]
        logger.info(f"Processing limited to {limit} images")
    
    # Initialize model
    logger.info("Loading SigLIP model...")
    model = get_model()
    
    # Initialize database
    logger.info("Connecting to Milvus...")
    db = get_database()
    
    # Reset collection if requested
    if reset:
        logger.warning("Resetting collection...")
        db.delete_collection()
        # Reinitialize database to recreate collection
        reset_database()
        db = get_database()
    
    # Step 3: Process images in batches
    total_images = len(image_files)
    total_inserted = 0
    total_skipped = 0
    
    # Batch data accumulators
    batch_ids: List[int] = []
    batch_paths: List[str] = []
    batch_categories: List[str] = []
    batch_genders: List[str] = []
    batch_colours: List[str] = []
    batch_seasons: List[str] = []
    batch_usages: List[str] = []
    batch_names: List[str] = []
    batch_embeddings: List[List[float]] = []
    
    logger.info(f"Starting ingestion of {total_images} images...")
    
    for idx, image_path in enumerate(tqdm(image_files, desc="Processing images")):
        try:
            # Parse ID from filename
            image_id = parse_image_id(image_path.name)
            
            if image_id is None:
                total_skipped += 1
                continue
            
            # Lookup metadata
            meta = metadata_dict.get(image_id, {})
            category = meta.get('articleType', 'Unknown')
            gender = meta.get('gender', 'Unknown')
            base_colour = meta.get('baseColour', 'Unknown')
            season = meta.get('season', 'Unknown')
            usage = meta.get('usage', 'Unknown')
            product_name = meta.get('productDisplayName', 'Unknown')
            
            # Truncate productDisplayName if too long (max 512 chars)
            if len(product_name) > 500:
                product_name = product_name[:500]
            
            # Load and validate image
            img = Image.open(image_path).convert("RGB")
            
            # Extract embedding
            embedding = model.get_image_embedding(img)
            
            # Add to batch
            batch_ids.append(image_id)
            batch_paths.append(str(image_path))
            batch_categories.append(category)
            batch_genders.append(gender)
            batch_colours.append(base_colour)
            batch_seasons.append(season)
            batch_usages.append(usage)
            batch_names.append(product_name)
            batch_embeddings.append(embedding.tolist())
            
            # Insert batch when full
            if len(batch_paths) >= batch_size:
                db.insert(
                    image_ids=batch_ids,
                    image_paths=batch_paths,
                    categories=batch_categories,
                    genders=batch_genders,
                    base_colours=batch_colours,
                    seasons=batch_seasons,
                    usages=batch_usages,
                    product_names=batch_names,
                    embeddings=batch_embeddings
                )
                total_inserted += len(batch_paths)
                
                # Clear batch
                batch_ids = []
                batch_paths = []
                batch_categories = []
                batch_genders = []
                batch_colours = []
                batch_seasons = []
                batch_usages = []
                batch_names = []
                batch_embeddings = []
                
                # Progress log
                if total_inserted % 500 == 0:
                    logger.info(f"Processed {total_inserted}/{total_images} images...")
        
        except Exception as e:
            logger.debug(f"Error processing {image_path.name}: {e}")
            total_skipped += 1
            continue
    
    # Insert remaining items in last batch
    if batch_paths:
        db.insert(
            image_ids=batch_ids,
            image_paths=batch_paths,
            categories=batch_categories,
            genders=batch_genders,
            base_colours=batch_colours,
            seasons=batch_seasons,
            usages=batch_usages,
            product_names=batch_names,
            embeddings=batch_embeddings
        )
        total_inserted += len(batch_paths)
    
    # Step 4: Flush and finalize
    db.collection.flush()
    
    # Summary
    logger.info("=" * 50)
    logger.info("Ingestion Complete!")
    logger.info(f"  Total processed: {total_inserted}")
    logger.info(f"  Total skipped:   {total_skipped}")
    logger.info(f"  Collection size: {db.count()}")
    logger.info("=" * 50)
    
    return total_inserted


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest Fashion Product Images dataset into Milvus"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for insert operations (default: 50)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing collection before ingesting"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to process (for testing)"
    )
    
    args = parser.parse_args()
    
    try:
        count = ingest_data(
            batch_size=args.batch_size,
            reset=args.reset,
            limit=args.limit
        )
        
        if count > 0:
            logger.info("✅ Ingestion successful!")
            sys.exit(0)
        else:
            logger.warning("⚠️ No images were ingested")
            sys.exit(1)
            
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
