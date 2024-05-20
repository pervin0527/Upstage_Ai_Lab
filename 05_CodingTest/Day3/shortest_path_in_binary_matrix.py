from typing import List
from collections import deque

class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        n = len(grid)
        if grid[0][0] == 1 or grid[n-1][n-1] == 1:
            return -1

        offset_x = [-1, -1, 0, 1, 1, 1, 0, -1]
        offset_y = [0, -1, -1, -1, 0, 1, 1, 1]
        visited = [[False] * n for _ in range(n)]
        distances = [[float('inf')] * n for _ in range(n)]
        q = deque()

        q.append((0, 0))
        visited[0][0] = True
        distances[0][0] = 1

        while q:
            x, y = q.popleft()
            dist = distances[x][y]
            if x == n-1 and y == n-1:
                return dist

            for i in range(8):
                dx, dy = offset_x[i], offset_y[i]
                tx, ty = x + dx, y + dy

                if 0 <= tx < n and 0 <= ty < n and grid[tx][ty] == 0 and not visited[tx][ty]:
                    q.append((tx, ty))
                    visited[tx][ty] = True
                    distances[tx][ty] = dist + 1

        return -1

test = Solution()
print(test.shortestPathBinaryMatrix([[0, 0, 0], [1, 1, 0], [1, 1, 0]]))
print(test.shortestPathBinaryMatrix([[1, 0, 0], [1, 1, 0], [1, 1, 0]]))
