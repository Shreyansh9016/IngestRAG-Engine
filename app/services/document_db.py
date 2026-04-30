import threading
from typing import List, Dict, Tuple
from app.models.domain import DocItem, VectorItem
from app.algorithms.bruteforce import BruteForce
from app.algorithms.hnsw import HNSW
from app.utils.distance import cosine

class DocumentDB:
    def __init__(self):
        self.store: Dict[int, DocItem] = {}
        self.hnsw = HNSW(m=16, ef_build=200)
        self.bf = BruteForce()
        self.lock = threading.Lock()
        self.next_id = 1
        self.dims = 0

    def insert(self, title: str, text: str, emb: List[float]) -> int:
        with self.lock:
            if self.dims == 0:
                self.dims = len(emb)

            item = DocItem(id=self.next_id, title=title, text=text, emb=emb)
            self.next_id += 1
            self.store[item.id] = item

            vi = VectorItem(id=item.id, metadata=title, category="doc", emb=emb)
            self.hnsw.insert(vi, cosine)
            self.bf.insert(vi)
            return item.id

    def search(self, q: List[float], k: int, max_dist: float = 0.7) -> List[Tuple[float, DocItem]]:
        with self.lock:
            if not self.store:
                return []

            if len(self.store) < 10:
                raw = self.bf.knn(q, k, cosine)
            else:
                raw = self.hnsw.knn(q, k, 50, cosine)

            out = []
            for d, id_ in raw:
                if id_ in self.store and d <= max_dist:
                    out.append((d, self.store[id_]))
            return out

    def remove(self, id_: int) -> bool:
        with self.lock:
            if id_ not in self.store:
                return False
            del self.store[id_]
            self.hnsw.remove(id_)
            self.bf.remove(id_)
            return True

    def all(self) -> List[DocItem]:
        with self.lock:
            return list(self.store.values())

    def size(self) -> int:
        with self.lock:
            return len(self.store)

    def get_dims(self) -> int:
        return self.dims
