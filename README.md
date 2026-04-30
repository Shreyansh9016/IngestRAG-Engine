# IngestRAG-Engine
A fully functional Python implementation of a Vector Database and RAG engine using FastAPI, Hugging Face via the Inference API, and Groq.

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
