from dataclasses import dataclass
from typing import List

@dataclass
class VectorItem:
    id: int
    metadata: str
    category: str
    emb: List[float]

@dataclass
class DocItem:
    id: int
    title: str
    text: str
    emb: List[float]
