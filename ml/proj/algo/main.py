import sys

class Graph:

	def __init__(self, a_b):
		self.a = a_b
		self.pms = [[0 for column in range(a_b)] for row in range(a_b)]

	def prnt_mst(self, parent):
		tw = 0
		for inx in range(1, self.n):
			tw += self.pms[inx][parent[inx]]
		print(tw)

	def min_key(self, mnw, mset):
		min_w = sys.maxsize
		minwindex = 0
		for inx in range(self.a):
			if mset[inx] is False and mnw[inx] < min_w:
				minwindex = inx
				min_w = mnw[inx]
		return minwindex

	def prims_mst(self):
		int_max = sys.maxsize
		nmv = self.a
		mnw = [int_max] * nmv
		mnw[0] = 0
		bmst = [0] * nmv
		bmst[0] = -1
		mset = [False] * nmv
		for inx in range(nmv):
			min_dist_v = self.min_key(mnw, mset)
			mset[min_dist_v] = True
			for index_v in range(nmv):
				if mset[index_v] is False and 0 < self.pms[min_dist_v][index_v] < mnw[index_v]:
					mnw[index_v] = self.pms[min_dist_v][index_v]
					bmst[index_v] = min_dist_v
		self.prnt_mst(bmst)


if __name__ == '__main__':
	a, m = list(map(int, input().split()))
	g = Graph(a)
	for t in range(m):
		a1, b1, c1 = list(map(int, input().split()))
		g.pms[a1-1][b1-1] = c1
		g.pms[b1-1][a1-1] = c1
	g.prims_mst()
