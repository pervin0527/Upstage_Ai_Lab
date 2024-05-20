from typing import List
from collections import deque

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        n, m = len(grid), len(grid[0])

        visited = [[False] * m for _ in range(n)]
        offset_x = [0, 0, -1, 1]
        offset_y = [-1, 1, 0, 0]

        n_island = 0
        q = deque()
        for i in range(n):
            for j in range(m):
                if grid[i][j] == '1' and not visited[i][j]:
                    q.append((i, j))
                    visited[i][j] = True
                    n_island += 1

                    while q:
                        cx, cy = q.popleft()                        
                        for k in range(4):
                            nx = cx + offset_x[k]
                            ny = cy + offset_y[k]

                            if 0 <= nx < n and 0 <= ny < m and grid[nx][ny] == '1' and not visited[nx][ny]:
                                q.append((nx, ny))
                                visited[nx][ny] = True

        return n_island
    
test = Solution()
print(test.numIslands([["1","1","1","1","0"],
                       ["1","1","0","1","0"],
                       ["1","1","0","0","0"],
                       ["0","0","0","0","0"]]))
