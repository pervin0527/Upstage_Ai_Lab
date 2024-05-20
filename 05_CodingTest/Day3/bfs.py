from collections import deque

def isValid(r, c):
    return 0 <= r < row_len and 0 <= c < col_len and grid[r][c] == 1 and not visited[r][c]

def bfs(r, c):
    q = deque()
    q.append((r, c))
    visited[r][c] = True  # 시작 노드를 방문으로 표시

    while q:
        cr, cc = q.popleft()
        print(cr, cc)
        for i in range(len(dr)):
            tr = cr + dr[i]
            tc = cc + dc[i]

            if isValid(tr, tc):
                q.append((tr, tc))
                visited[tr][tc] = True  # 새로운 노드를 방문으로 표시

grid = [
   [1, 1, 1, 1],
   [0, 1, 0, 1],
   [0, 1, 0, 1],
   [1, 0, 1, 1],
]

row_len, col_len = len(grid), len(grid[0])
visited = [[False] * col_len for _ in range(row_len)]

dr = [0, 1, 0, -1]
dc = [1, 0, -1, 0]

bfs(0, 0)
