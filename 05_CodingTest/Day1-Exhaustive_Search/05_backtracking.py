from typing import List

nums = [4, 9, 7, 5, 1]
k = 4
target = 21

def backtrack(start: int, curr_arr: List[int]):
    if len(curr_arr) == k:
        total = sum(nums[item] for item in curr_arr)
        if total == target:
            return curr_arr[:]

    for i in range(start, len(nums)):
        curr_arr.append(i)
        result = backtrack(i + 1, curr_arr)
        if result:
            return result
        curr_arr.pop()

result = backtrack(0, [])
print(result)