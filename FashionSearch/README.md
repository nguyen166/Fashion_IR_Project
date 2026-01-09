# FashionSearch

Fashion Image Retrieval System with SigLIP embeddings and Rocchio Relevance Feedback.

## 🌟 Features

- **Text-to-Image Search**: Find fashion items by describing them in natural language
- **Image-to-Image Search**: Upload an image to find visually similar items
- **Relevance Feedback**: Improve search results using Rocchio Algorithm based on user feedback

## 🏗️ Architecture

This is a monolithic FastAPI application that combines:
- **SigLIP Model** (`google/siglip-base-patch16-224`) for image/text embeddings
- **Milvus** for vector similarity search
- **Rocchio Algorithm** for relevance feedback

## 📁 Project Structure

```
FashionSearch/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application & endpoints
│   ├── config.py            # Configuration settings
│   ├── schemas.py           # Pydantic models
│   ├── core/
│   │   ├── model.py         # SigLIP model wrapper
│   │   └── database.py      # Milvus database operations
│   └── services/
│       └── search.py        # Search logic & Rocchio algorithm
├── data/                    # Image dataset directory
├── ingest.py               # Data ingestion script
├── requirements.txt
├── docker-compose.yml
└── .env
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.9+
- Docker & Docker Compose (for Milvus)
- CUDA-capable GPU (optional, for faster processing)

### 2. Setup Milvus

```bash
# Start Milvus using Docker Compose
docker-compose up -d
```

### 3. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install packages
pip install -r requirements.txt
```

### 4. Prepare Dataset

Download the **Fashion Product Images (Small)** dataset from Kaggle and organize in `data/`:

```
data/
├── images/         # All images (1000.jpg, 1001.jpg, ...)
└── styles.csv      # Metadata file
```

### 5. Ingest Data

```bash
# Run ingestion script
python ingest.py --reset

# Or with options
python ingest.py --batch-size 50 --limit 1000 --reset
```

### 6. Start API Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 📖 API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Endpoints

#### Health Check
```bash
GET /health
```

#### Text Search
```bash
POST /search/text
Content-Type: application/json

{
    "query": "red floral dress",
    "top_k": 10,
    "category": null
}
```

#### Image Search
```bash
POST /search/image
Content-Type: multipart/form-data

file: <image_file>
top_k: 10
category: null
```

#### Relevance Feedback Search
```bash
POST /search/feedback
Content-Type: application/json

{
    "query_vector": [...],      # From previous search response
    "positive_ids": [1, 5, 8],  # IDs of relevant results
    "negative_ids": [3, 7],     # IDs of non-relevant results
    "top_k": 10,
    "alpha": 1.0,               # Weight for original query
    "beta": 0.75,               # Weight for positive feedback
    "gamma": 0.25               # Weight for negative feedback
}
```

## 🔬 Rocchio Algorithm

The relevance feedback uses the Rocchio Algorithm:

$$Q_{new} = \alpha \cdot Q_{old} + \beta \cdot \frac{1}{|D_r|} \sum_{d \in D_r} d - \gamma \cdot \frac{1}{|D_{nr}|} \sum_{d \in D_{nr}} d$$

Where:
- $Q_{old}$: Original query vector
- $D_r$: Set of relevant (positive) document vectors
- $D_{nr}$: Set of non-relevant (negative) document vectors
- $\alpha, \beta, \gamma$: Tunable weights

**Default Parameters:**
- $\alpha = 1.0$ (original query weight)
- $\beta = 0.75$ (positive feedback weight)
- $\gamma = 0.25$ (negative feedback weight)

## ⚙️ Configuration

Edit `.env` file or set environment variables:

```env
# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=fashion_search

# Model
MODEL_NAME=google/siglip-base-patch16-224
EMBEDDING_DIM=768

# Search
DEFAULT_TOP_K=10

# Rocchio Parameters
ROCCHIO_ALPHA=1.0
ROCCHIO_BETA=0.75
ROCCHIO_GAMMA=0.25
```

## 🐳 Docker Deployment

### Full Stack Deployment

```bash
# Build and start all services
docker-compose up -d

# Check logs
docker-compose logs -f
```

## 📊 Example Workflow

1. **Initial Search** (Text or Image)
   ```python
   response = requests.post("/search/text", json={"query": "blue jeans"})
   results = response.json()["results"]
   query_vector = response.json()["query_vector"]
   ```

2. **User Reviews Results**
   - Mark result IDs 1, 3, 7 as relevant
   - Mark result IDs 2, 5 as not relevant

3. **Feedback Search**
   ```python
   response = requests.post("/search/feedback", json={
       "query_vector": query_vector,
       "positive_ids": [1, 3, 7],
       "negative_ids": [2, 5],
       "top_k": 10
   })
   refined_results = response.json()["results"]
   ```

4. **Repeat** steps 2-3 for better results

## 📝 License

MIT License
