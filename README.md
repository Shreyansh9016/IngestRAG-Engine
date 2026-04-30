# IngestRAG-Engine
A fully functional Python implementation of a Vector Database and RAG engine using FastAPI, Hugging Face via the Inference API, and Groq.

## What This Project Does

| Feature | Description |
|---|---|
| **3 Search Algorithms** | HNSW (production-grade), KD-Tree, Brute Force — run all three and compare speed |
| **3 Distance Metrics** | Cosine similarity, Euclidean distance, Manhattan distance |
| **16D Demo Vectors** | 20 pre-loaded semantic vectors across 4 categories (CS, Math, Food, Sports) |
| **2D PCA Scatter Plot** | Live visualization of semantic space — watch clusters form |
| **Real Document Embedding** | Paste any text → Groq embeds it  |
| **RAG Pipeline** | Ask questions about your documents → HNSW retrieves context → local LLM answers |
| **Full REST API** | CRUD endpoints: insert, delete, search, benchmark, hnsw-info |

## How It Works

```
Your Text
    │
    ▼
  Groq                                ← converts text to a 768-dimensional vector
    │
    ▼
HNSW Index (C++)                   ← indexes the vector in a multilayer graph
    │
    ▼
Semantic Search                    ← finds nearest neighbors in vector space
    │
    ▼
Groq (instant)                  ← reads retrieved chunks, generates an answer
    │
    ▼
Answer
```

**HNSW (Hierarchical Navigable Small World)** is the same algorithm used by Pinecone, Weaviate, Chroma, and Milvus. It builds a multilayer graph where each layer is progressively sparser — searches start at the top layer and zoom in, achieving O(log N) complexity instead of O(N) for brute force.

---

## Requirements
- Python 3.9+
- Install dependencies: `pip install -r requirements.txt`

## Configuration
This project uses a `.env` file for API key management. Before running the server, create a `.env` file in the project root directory and add your keys:

```env
GROQ_API_KEY="your_groq_api_key_here"
HF_API_KEY="your_huggingface_api_key_here_optional"
```

*Note: Groq is required for text generation via the local LLM integration. Hugging Face is used for embeddings.*

## Running the Server
Start the FastAPI server via Uvicorn from the project root directory:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

## Usage
Once the server is running, visit [http://localhost:8080](http://localhost:8080) to interact with the frontend UI.
You can view the detailed documentation on how the vector indexing algorithms and features work in `usage.md`.
