def isValid(r, c):
    return 0 <= r < row_len and 0 <= c < col_len and grid[r][c] == 1 and not visited[r][c]

def dfs(r, c):
    visited[r][c] = True
    print(r, c)

    for i in range(len(dr)):
        tr, tc = r + dr[i], c + dc[i]
        
        if isValid(tr, tc):
            dfs(tr, tc)


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

dfs(0, 0)