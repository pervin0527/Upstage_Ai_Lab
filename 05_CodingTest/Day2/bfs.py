from collections import deque

def bfs(graph, start_v):
    ## 처음 시작할 정점.
    q = deque()
    q.append(start_v)
    visited = {start_v : True}

    while q:
        cur_v = q.popleft()
        print(cur_v, end=' ')
        for next_v in graph[cur_v]:
            if next_v not in visited:
                q.append(next_v) ## 예약
                visited[next_v] = True ## 방문 표시


graph = {
    0 : [1, 3, 6],
    1 : [0, 3],
    2 : [3],
    3 : [0, 1, 2, 7],
    4 : [5],
    5 : [4, 6, 7],
    6 : [0, 5],
    7 : [3, 5],
}

bfs(graph, start_v=0)