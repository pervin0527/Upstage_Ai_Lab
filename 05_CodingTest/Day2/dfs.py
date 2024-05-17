def dfs(graph, start_v):
    stack = [start_v]
    visited = {start_v: True}

    while stack:
        cur_v = stack.pop()
        print(cur_v, end=' ')
        # for next_v in graph[cur_v]:

        ## stack은 후입선출이기 때문에 reversed를 하지 않으면 늦게 저장된 인접 노드를 먼저 방문함.
        for next_v in reversed(graph[cur_v]):
            if next_v not in visited:
                stack.append(next_v)
                visited[next_v] = True


def dfs_recur(cur_v):
    visited[cur_v] = True
    print(cur_v, end=' ')

    for next_v in graph[cur_v]:
        if next_v not in visited:
            dfs_recur(next_v)



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

dfs(graph, start_v=0)
print()

visited = {}
dfs_recur(0)