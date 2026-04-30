from typing import List, Tuple
from app.models.domain import VectorItem
from app.utils.distance import DistFn

class BruteForce:
    def __init__(self):
        self.items: List[VectorItem] = []

    def insert(self, v: VectorItem):
        self.items.append(v)

    def knn(self, q: List[float], k: int, dist: DistFn) -> List[Tuple[float, int]]:
        r = [(dist(q, v.emb), v.id) for v in self.items]
        r.sort(key=lambda x: x[0])
        return r[:k]

    def remove(self, id: int):
        self.items = [v for v in self.items if v.id != id]
