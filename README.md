<h1>🚀 IngestRAG-Engine</h1>

<p>
A fully functional Python implementation of a <b>Vector Database + RAG Engine</b> 
using <b>FastAPI</b>, <b>Groq</b>, and <b>Hugging Face Inference API</b>.
</p>

<hr>

<h2>✨ Features</h2>

<table>
<tr><th>Feature</th><th>Description</th></tr>
<tr>
<td><b>3 Search Algorithms</b></td>
<td>HNSW (production-grade), KD-Tree, Brute Force — compare performance</td>
</tr>
<tr>
<td><b>3 Distance Metrics</b></td>
<td>Cosine Similarity, Euclidean Distance, Manhattan Distance</td>
</tr>
<tr>
<td><b>16D Demo Vectors</b></td>
<td>20 semantic vectors across CS, Math, Food, Sports</td>
</tr>
<tr>
<td><b>2D PCA Visualization</b></td>
<td>Interactive scatter plot showing semantic clustering</td>
</tr>
<tr>
<td><b>Real Embedding</b></td>
<td>Convert real text → embeddings using Groq</td>
</tr>
<tr>
<td><b>RAG Pipeline</b></td>
<td>Query your documents → retrieve context → generate answer</td>
</tr>
<tr>
<td><b>REST API</b></td>
<td>Insert, delete, search, benchmark, HNSW info</td>
</tr>
</table>

<hr>

<h2>⚙️ How It Works</h2>

<pre>
Your Text
    │
    ▼
Groq (Embedding)
    │
    ▼
HNSW Index (Graph-based search)
    │
    ▼
Semantic Search (Nearest Neighbors)
    │
    ▼
Groq (LLM Answer Generation)
    │
    ▼
Final Answer
</pre>

<p>
<b>HNSW (Hierarchical Navigable Small World)</b> builds a multi-layer graph 
to achieve <b>O(log N)</b> search complexity — far faster than brute force.
</p>

<hr>

<h2>🧰 Requirements</h2>

<ul>
<li>Python 3.9+</li>
<li>pip (Python package manager)</li>
</ul>

<hr>

<h2>🔧 Setup Guide (Step-by-Step)</h2>

<h3>1️⃣ Clone the Repository</h3>

<pre><code>git clone https://github.com/your-username/IngestRAG-Engine.git
cd IngestRAG-Engine</code></pre>

<hr>

<h3>2️⃣ Create Virtual Environment</h3>

<p><b>Windows:</b></p>
<pre><code>python -m venv venv</code></pre>

<p><b>Mac/Linux:</b></p>
<pre><code>python3 -m venv venv</code></pre>

<hr>

<h3>3️⃣ Activate Virtual Environment</h3>

<p><b>Windows (PowerShell / CMD):</b></p>
<pre><code>venv\Scripts\activate</code></pre>

<p><b>Mac/Linux:</b></p>
<pre><code>source venv/bin/activate</code></pre>

<p>
After activation, your terminal should show:
<code>(venv)</code>
</p>

<hr>

<h3>4️⃣ Install Dependencies</h3>

<pre><code>pip install -r requirements.txt</code></pre>

<hr>

<h3>5️⃣ Set Environment Variables</h3>

<p>Create a <code>.env</code> file in the root directory:</p>

<pre><code>GROQ_API_KEY="your_groq_api_key_here"
HF_API_KEY="your_huggingface_api_key_here_optional"</code></pre>

<p>
<b>Note:</b><br>
Groq → required for LLM responses<br>
Hugging Face → used for embeddings (optional fallback)
</p>

<hr>

<h3>6️⃣ Run the Server</h3>

<pre><code>uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload</code></pre>

<hr>

<h2>🌐 Usage</h2>

<p>
Once running, open:
</p>

<p>
👉 <a href="http://localhost:8080">http://localhost:8080</a>
</p>

<ul>
<li>Use the UI for semantic search & RAG queries</li>
<li>Test APIs via Swagger docs</li>
</ul>

<hr>

<h2>📡 API Endpoints</h2>

<ul>
<li><code>/insert</code> → Add vectors</li>
<li><code>/search</code> → Query nearest neighbors</li>
<li><code>/delete</code> → Remove vectors</li>
<li><code>/benchmark</code> → Compare algorithms</li>
<li><code>/hnsw-info</code> → View index structure</li>
</ul>

<hr>

<h2>📊 Tech Stack</h2>

<ul>
<li>FastAPI</li>
<li>Groq LLM</li>
<li>Hugging Face Inference API</li>
<li>NumPy / SciPy</li>
<li>Custom HNSW Implementation</li>
</ul>

<hr>

<h2>📌 Future Improvements</h2>

<ul>
<li>Docker deployment</li>
<li>Cloud vector DB integration</li>
<li>Authentication & multi-user support</li>
<li>Streaming responses</li>
</ul>

<hr>

<h2>🤝 Contributing</h2>

<p>
Pull requests are welcome! Feel free to fork and improve.
</p>

<hr>

<h2>📜 License</h2>

<p>
MIT License
</p>

<hr>

<h3>⭐ If you like this project, give it a star on GitHub!</h3>
