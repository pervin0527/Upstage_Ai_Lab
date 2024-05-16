from typing import List

class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        answer = []
        def recur(st, arr):
            if len(arr) == k:
                answer.append(arr[:])
                return
            
            for i in range(st, n+1):
                arr.append(i)
                recur(i+1, arr)
                arr.pop()
        
        recur(1, [])
        return answer
    
    def permut(self, n: int, k: int) -> List[List[int]]:
        answer = []
        def recur(st, arr):
            if len(arr) == k:
                answer.append(arr[:])
                return
            
            for i in range(1, n+1):
                arr.append(i)
                recur(i+1, arr)
                arr.pop()
        
        recur(1, [])
        return answer


test = Solution()
print(test.combine(4, 2))
print(test.permut(4, 2))