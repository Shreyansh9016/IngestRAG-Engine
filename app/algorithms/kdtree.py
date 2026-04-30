from typing import List, Tuple, Optional
import heapq
from app.models.domain import VectorItem
from app.utils.distance import DistFn

class KDNode:
    def __init__(self, item: VectorItem):
        self.item = item
        self.left: Optional['KDNode'] = None
        self.right: Optional['KDNode'] = None

class KDTree:
    def __init__(self, dims: int):
        self.root: Optional[KDNode] = None
        self.dims = dims

    def _ins(self, n: Optional[KDNode], v: VectorItem, d: int) -> KDNode:
        if n is None:
            return KDNode(v)
        ax = d % self.dims
        if v.emb[ax] < n.item.emb[ax]:
            n.left = self._ins(n.left, v, d + 1)
        else:
            n.right = self._ins(n.right, v, d + 1)
        return n

    def insert(self, v: VectorItem):
        self.root = self._ins(self.root, v, 0)

    def _knn(self, n: Optional[KDNode], q: List[float], k: int, d: int, dist: DistFn, heap: List[Tuple[float, int]]):
        if not n:
            return

        dn = dist(q, n.item.emb)
        # heap is a max-heap (using negative distance)
        # we store (-dn, id) in heapq
        if len(heap) < k or dn < -heap[0][0]:
            heapq.heappush(heap, (-dn, n.item.id))
            if len(heap) > k:
                heapq.heappop(heap)

        ax = d % self.dims
        diff = q[ax] - n.item.emb[ax]

        closer = n.left if diff < 0 else n.right
        farther = n.right if diff < 0 else n.left

        self._knn(closer, q, k, d + 1, dist, heap)

        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, q, k, d + 1, dist, heap)

    def knn(self, q: List[float], k: int, dist: DistFn) -> List[Tuple[float, int]]:
        heap = []
        self._knn(self.root, q, k, 0, dist, heap)

        # Convert back to positive distance and sort
        r = [(-mdn, id) for mdn, id in heap]
        r.sort(key=lambda x: x[0])
        return r

    def rebuild(self, items: List[VectorItem]):
        self.root = None
        for v in items:
            self.insert(v)
