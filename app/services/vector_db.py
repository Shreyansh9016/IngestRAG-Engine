import time
import threading
from typing import List, Dict, Any
from app.models.domain import VectorItem
from app.algorithms.bruteforce import BruteForce
from app.algorithms.kdtree import KDTree
from app.algorithms.hnsw import HNSW
from app.utils.distance import get_dist_fn

class VectorDB:
    def __init__(self, dims: int):
        self.dims = dims
        self.store: Dict[int, VectorItem] = {}
        self.bf = BruteForce()
        self.kdt = KDTree(dims)
        self.hnsw = HNSW(m=16, ef_build=200)
        self.lock = threading.Lock()
        self.next_id = 1

    def insert(self, meta: str, cat: str, emb: List[float], dist_metric: str = "cosine") -> int:
        dist = get_dist_fn(dist_metric)
        with self.lock:
            v = VectorItem(id=self.next_id, metadata=meta, category=cat, emb=emb)
            self.next_id += 1
            self.store[v.id] = v
            self.bf.insert(v)
            self.kdt.insert(v)
            self.hnsw.insert(v, dist)
            return v.id

    def remove(self, id_: int) -> bool:
        with self.lock:
            if id_ not in self.store:
                return False
            del self.store[id_]
            self.bf.remove(id_)
            self.hnsw.remove(id_)
            rem = list(self.store.values())
            self.kdt.rebuild(rem)
            return True

    def search(self, q: List[float], k: int, metric: str, algo: str) -> Dict[str, Any]:
        with self.lock:
            dfn = get_dist_fn(metric)
            t0 = time.time()

            if algo == "bruteforce":
                raw = self.bf.knn(q, k, dfn)
            elif algo == "kdtree":
                raw = self.kdt.knn(q, k, dfn)
            else:
                raw = self.hnsw.knn(q, k, 50, dfn)

            us = int((time.time() - t0) * 1_000_000)

            hits = []
            for d, id_ in raw:
                item = self.store.get(id_)
                if item:
                    hits.append({
                        "id": id_,
                        "metadata": item.metadata,
                        "category": item.category,
                        "embedding": item.emb,
                        "distance": d
                    })

            return {
                "results": hits,
                "latencyUs": us,
                "algo": algo,
                "metric": metric
            }

    def benchmark(self, q: List[float], k: int, metric: str) -> Dict[str, Any]:
        with self.lock:
            dfn = get_dist_fn(metric)

            def time_it(fn) -> int:
                t0 = time.time()
                fn()
                return int((time.time() - t0) * 1_000_000)

            bf_us = time_it(lambda: self.bf.knn(q, k, dfn))
            kd_us = time_it(lambda: self.kdt.knn(q, k, dfn))
            hnsw_us = time_it(lambda: self.hnsw.knn(q, k, 50, dfn))

            return {
                "bruteforceUs": bf_us,
                "kdtreeUs": kd_us,
                "hnswUs": hnsw_us,
                "itemCount": len(self.store)
            }

    def all(self) -> List[VectorItem]:
        with self.lock:
            return list(self.store.values())

    def hnsw_info(self) -> Dict[str, Any]:
        with self.lock:
            return self.hnsw.get_info()

    def size(self) -> int:
        with self.lock:
            return len(self.store)
