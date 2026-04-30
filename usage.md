# IngestRAG-Engine Vector Database Usage Guide

Welcome to the **IngestRAG-Engine** Vector Engine interface! This document covers each feature of the frontend, explaining how to use it and detailing the underlying mathematical, algorithmic, and architectural mechanics occurring inside the Python backend.

---

## 🧭 Left Panel: Control & Insert 

### 1. Query (Demo Vectors)
- **How to use**: Type a specific term like *"binary tree"* or a hybrid concept like *"sushi mathematics"* and click `SEARCH`.
- **Technical mechanics**: This text is converted into a deterministic `16-Dimensional` categorical embedding using a semantic heuristic function (scoring against hardcoded vocabulary for CS, Math, Food, Sports). The backend then takes this 16D array and queries the database using the selected index algorithm and metric. 

### 2. Algorithm
- **How to use**: Select the backend index used for nearest neighbor retrievals: HNSW, KD-Tree, or Brute Force.
- **Technical mechanics**:
    - **Brute Force**: Iterates linearly ($O(N)$) calculating the distance between the query and *every* vector in the database, tracking the top-k in a max heap. Slowest, but $100\%$ accurate (ground truth).
    - **KD-Tree**: Navigates down a spatial partitioning tree, pruning branches where the distance boundaries prove smaller distances aren't mathematically possible. Fast ($O(\log N)$) but loses accuracy rapidly in dimensions $>20$.
    - **HNSW (Hierarchical Navigable Small World)**: A state-of-the-art multilayer graph. Starts at the top sparse layer to find the regional neighborhood, drills down, and explores connections linearly. Hyper-fast ($O(\log N)$) and scales beautifully for dense vectors.

### 3. Distance Metric
- **How to use**: Choose how "similarity" is mathematically measured between two point arrays.
- **Technical mechanics**:
    - **Cosine Similarity**: Measures the *angle* between two vectors. Great for semantic text similarity.
    - **Euclidean (L2)**: Measures the straight-line physical distance. Great for spatial data.
    - **Manhattan (L1)**: Measures grid-like steps across axes.

### 4. Benchmark: Compare All Algos
- **How to use**: Runs a query across all 3 algorithms simultaneously to compare execution speed.
- **Technical mechanics**: The FastAPI backend routes to `/benchmark`, firing the same query concurrently into the python classes for BruteForce, KDTree, and HNSW. It tracks `time.time()` latency for each (in microseconds), proving that HNSW operates vastly faster than flat brute-force as datasets scale.

---

## 📈 Center Panel: Visualizer

### 2D PCA Scatter Plot
- **How to use**: Watch the dots dance. Clicking search highlights hits, querying moves the star. 
- **Technical mechanics**: The 16-Dimensional (or 384-dimensional) embedding space cannot be rendered on a 2D monitor. The UI continuously runs **PCA (Principal Component Analysis)** via Power Iteration to crush the N-Dimensional vectors down to 2 axes (PC1, PC2) that capture the maximum statistical variance.

---

## ⚙️ Right Panel: Results & RAG

### SEARCH Tab
- **Stats**: Shows the true microsecond latency returned by the backend `app.services.vector_db`.
- **Top Matches**: Renders the Top-K elements (using a Max Heap from Python's `heapq`).
- **HNSW Graph Layers**: Introspects the actual nodes distribution across the graph layers inside `hnsw.py`.

### DOCUMENTS Tab
- **How to use**: Paste a paragraph of text and click `EMBED & INSERT`.
- **Technical mechanics**:
    1. **Chunking**: The backend (`chunker.py`) slices the text into overlapping pieces (e.g., 250 words per chunk).
    2. **Embedding**: `llm_client.py` makes a POST securely to the Hugging Face Serverless Inference API, pushing the text through `sentence-transformers/all-MiniLM-L6-v2`.
    3. **Indexing**: It receives a 384-dimensional vector, linking the text chunk, and inserts it into the `DocumentDB` graph.

### ASK AI Tab (RAG)
- **How to use**: Type a question (e.g., *"What is HNSW?"*).
- **Technical mechanics (Retrieval Augmented Generation)**:
    1. **Query Embed**: Hugging Face embeds your question into a 384D vector.
    2. **Vector Search**: The DocumentDB finds the Top-K nearest text chunks.
    3. **Prompt Injection**: The Groq client injects those contexts inside a system prompt wrapper.
    4. **Inference**: Groq evaluates the prompt using the `llama-3.1-8b-instant` model and streams back a naturally synthesized answer grounded exclusively on the retrieved facts.
