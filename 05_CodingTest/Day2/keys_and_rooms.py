"""
n개의 방이 있습니다. 방들은 0부터 n-1까지 번호가 매겨져 있습니다.
모든 방은 잠겨 있지만, 방 0은 열려 있습니다.
목표는 모든 방을 방문하는 것입니다.
하지만 방의 열쇠를 가지지 않으면 잠긴 방에 들어갈 수 없습니다.
방을 방문하면, 그 방에서 얻을 수 있는 고유한 열쇠들의 집합을 찾을 수 있습니다. 각 열쇠에는 어떤 방을 열 수 있는지 번호가 적혀 있으며, 이 열쇠들을 모두 가져가 다른 방들을 열 수 있습니다.
배열 rooms가 주어졌을 때, rooms[i]는 방 i를 방문하면 얻을 수 있는 열쇠들의 집합을 나타냅니다.
"""

from typing import List
from collections import deque

class Solution:
    ## BFS
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        q = deque()
        visited = [False] * len(rooms)

        q.append(0)
        visited[0] = True

        while q:
            room_number = q.popleft()
            keys = rooms[room_number]
            for key in keys:
                if not visited[key]:
                    q.append(key)
                    visited[key] = True

        # print(visited)
        return all(visited)
    
    ## DFS
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        visited = [False] * len(rooms)

        def dfs(curr):
            visited[curr] = True

            for key in rooms[curr]:
                if not visited[key]:
                    dfs(key)

        dfs(0)
        return all(visited)