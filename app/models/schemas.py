from pydantic import BaseModel
from typing import List

class InsertReq(BaseModel):
    metadata: str
    category: str
    embedding: List[float]

class DocInsertReq(BaseModel):
    title: str
    text: str

class DocSearchReq(BaseModel):
    question: str
    k: int = 3
