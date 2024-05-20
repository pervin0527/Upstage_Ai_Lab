from typing import List
from collections import deque

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        n, m = len(grid), len(grid[0])

        n_island = 0
        visited = [[False] * m for _ in range(n)]
        offset_x = [0, 0, -1, 1]
        offset_y = [-1, 1, 0, 0]

        def dfs(x, y):
            visited[x][y] = True

            for i in range(4):
                dx, dy = offset_x[i], offset_y[i]
                tx, ty = x + dx, y + dy

                if 0 <= tx < n and 0 <= ty < m and grid[tx][ty] == '1' and not visited[tx][ty]:
                    dfs(tx, ty)

        for i in range(n):
            for j in range(m):
                if grid[i][j] == '1' and not visited[i][j]:
                    dfs(i, j)
                    n_island += 1

        return n_island
    
test = Solution()
print(test.numIslands([["1","1","1","1","0"],
                       ["1","1","0","1","0"],
                       ["1","1","0","0","0"],
                       ["0","0","0","0","0"]]))
