from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path

# 🔹 Get base directory safely
BASE_DIR = Path(__file__).resolve().parent.parent

# 🔹 Load env only locally (optional safety)
if os.path.exists(BASE_DIR / ".env"):
    from dotenv import load_dotenv
    load_dotenv()

from app.api.endpoints import router

app = FastAPI(title="VectorDB Engine (Python)")

# 🔹 CORS (allow frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# 🔹 Include API routes
app.include_router(router)

# 🔹 Serve static files (if you add CSS/JS later)
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")

# 🔹 Serve index.html safely
@app.get("/", response_class=HTMLResponse)
def read_index():
    index_path = BASE_DIR / "index.html"
    
    if index_path.exists():
        return FileResponse(index_path)
    
    return "<h1>index.html not found</h1>"