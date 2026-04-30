from typing import List, Tuple, Dict, Any
import math
import heapq
import random
from app.models.domain import VectorItem
from app.utils.distance import DistFn

class Node:
    def __init__(self, item: VectorItem, max_lyr: int):
        self.item = item
        self.max_lyr = max_lyr
        self.nbrs: List[List[int]] = [[] for _ in range(max_lyr + 1)]

class HNSW:
    def __init__(self, m: int = 16, ef_build: int = 200):
        self.G: Dict[int, Node] = {}
        self.M = m
        self.M0 = 2 * m
        self.ef_build = ef_build
        self.m_L = 1.0 / math.log(m)
        self.top_layer = -1
        self.entry_pt = -1
        random.seed(42)  # Maintain deterministic nature if possible

    def _rand_level(self) -> int:
        u = random.uniform(0.0, 1.0)
        return int(math.floor(-math.log(u) * self.m_L))

    def _search_layer(self, q: List[float], ep: int, ef: int, lyr: int, dist: DistFn) -> List[Tuple[float, int]]:
        vis: Dict[int, bool] = {}
        # Min heap for candidates, storing (distance, item_id)
        cands = []
        # Max heap for found elements, storing (-distance, item_id)
        found = []

        d0 = dist(q, self.G[ep].item.emb)
        vis[ep] = True
        heapq.heappush(cands, (d0, ep))
        heapq.heappush(found, (-d0, ep))

        while cands:
            cd, cid = heapq.heappop(cands)
            # If closest candidate is further than furthest found, break
            if len(found) >= ef and cd > -found[0][0]:
                break

            if lyr >= len(self.G[cid].nbrs):
                continue

            for nid in self.G[cid].nbrs[lyr]:
                if vis.get(nid) or nid not in self.G:
                    continue
                vis[nid] = True
                nd = dist(q, self.G[nid].item.emb)

                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        res = [(-nd, nid) for nd, nid in found]
        res.sort(key=lambda x: x[0])
        return res

    def _select_nbrs(self, cands: List[Tuple[float, int]], max_m: int) -> List[int]:
        return [nid for _, nid in cands[:max_m]]

    def insert(self, item: VectorItem, dist: DistFn):
        id_ = item.id
        lvl = self._rand_level()
        self.G[id_] = Node(item, lvl)

        if self.entry_pt == -1:
            self.entry_pt = id_
            self.top_layer = lvl
            return

        ep = self.entry_pt
        for lc in range(self.top_layer, lvl, -1):
            if lc < len(self.G[ep].nbrs):
                W = self._search_layer(item.emb, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]

        for lc in range(min(self.top_layer, lvl), -1, -1):
            W = self._search_layer(item.emb, ep, self.ef_build, lc, dist)
            max_m = self.M0 if lc == 0 else self.M
            sel = self._select_nbrs(W, max_m)
            self.G[id_].nbrs[lc] = sel

            for nid in sel:
                if nid not in self.G:
                    continue
                if len(self.G[nid].nbrs) <= lc:
                    self.G[nid].nbrs.extend([[] for _ in range(lc + 1 - len(self.G[nid].nbrs))])

                conn = self.G[nid].nbrs[lc]
                conn.append(id_)
                if len(conn) > max_m:
                    ds = []
                    for c in conn:
                        if c in self.G:
                            ds.append((dist(self.G[nid].item.emb, self.G[c].item.emb), c))
                    ds.sort(key=lambda x: x[0])
                    self.G[nid].nbrs[lc] = [c for _, c in ds[:max_m]]

            if W:
                ep = W[0][1]

        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt = id_

    def knn(self, q: List[float], k: int, ef: int, dist: DistFn) -> List[Tuple[float, int]]:
        if self.entry_pt == -1:
            return []
        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            if lc < len(self.G[ep].nbrs):
                W = self._search_layer(q, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]

        W = self._search_layer(q, ep, max(ef, k), 0, dist)
        return W[:k]

    def remove(self, id_: int):
        if id_ not in self.G:
            return

        for nid, nd in self.G.items():
            for layer in nd.nbrs:
                if id_ in layer:
                    layer.remove(id_)

        if self.entry_pt == id_:
            self.entry_pt = -1
            for nid, nd in self.G.items():
                if nid != id_:
                    self.entry_pt = nid
                    break
        del self.G[id_]

    def get_info(self) -> Dict[str, Any]:
        top_layer = self.top_layer
        node_count = len(self.G)
        max_l = max(top_layer + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
        nodes = []
        edges = []

        for id_, nd in self.G.items():
            nodes.append({
                "id": id_,
                "metadata": nd.item.metadata,
                "category": nd.item.category,
                "maxLyr": nd.max_lyr
            })
            for lc in range(min(nd.max_lyr + 1, max_l)):
                nodes_per_layer[lc] += 1
                if lc < len(nd.nbrs):
                    for nid in nd.nbrs[lc]:
                        if id_ < nid:
                            edges_per_layer[lc] += 1
                            edges.append({"src": id_, "dst": nid, "lyr": lc})

        return {
            "topLayer": top_layer,
            "nodeCount": node_count,
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes": nodes,
            "edges": edges
        }

    def __len__(self) -> int:
        return len(self.G)
