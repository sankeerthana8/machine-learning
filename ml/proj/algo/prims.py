import sys

class Graph:

    def __init__(self,a_b):
        self.no = a_b
        self.graph = [[0 for column in range(a_b)] for row in range(a_b)]

    def print_mst(self, parent):
        t_w = 0
        for inx in range(1, self.no):
            t_w = t_w + self.graph[inx][parent[inx]]
            k=t_w
        print(k)

    def min_key(self, mnw, mset):
        mw = sys.maxsize
        minx = 0
        boolean=False
        for inx in range(self.no):
            if mset[inx] is boolean:
                if mnw[inx] < mw:
                    minx = inx
                    mw = mnw[inx]
        return minx

    def prims_MST(self):  
        max = sys.maxsize
        nv = self.no
        boolf=False
        boolt=True
        mnw = [max] * nv
        mnw[0] = 0
        bmst = [0] * nv
        bmst[0] = -1
        mset = [boolf] * nv
        for inx in range(nv):
            min_v = self.min_key(mnw, mset)
            mset[min_v] = boolt
            for inx_v in range(nv):
                if mset[inx_v] is boolf:
                    if 0 < self.graph[min_v][inx_v] < mnw[inx_v]:
                        mnw[inx_v] = self.graph[min_v][inx_v]
                        bmst[inx_v] = min_v
        self.print_mst(bmst)


if __name__ == '__main__':
    x, y = list(map(int, input().split()))
    gr = Graph(x)
    for i in range(y):
        p, H, w = list(map(int, input().split()))
        gr.graph[p-1][H-1] = w
        gr.graph[H-1][p-1] = w
    gr.prims_MST()
