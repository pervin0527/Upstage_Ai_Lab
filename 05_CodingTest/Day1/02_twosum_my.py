from typing import List

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if nums[i] + nums[j] == target:
                    return [i, j]
                
    def hash(self, nums: List[int], target: int):
        num_to_index = {}
        for i, num in enumerate(nums):
            complement = target - num ## 각 요소에 대해 target에서 현재 요소를 뺀 값을 계산하여 complement를 구함.

            ## complement가 이미 해시맵에 있는지 확인
            if complement in num_to_index:
                return [num_to_index[complement], i] ## complement가 해시맵에 있다면, 두 숫자의 합이 target과 같으므로 그 인덱스를 반환

            num_to_index[num] = i ## 그렇지 않다면, 현재 숫자와 그 인덱스를 해시맵에 저장.

    def twopointer(self, nums: List[int], target: int):
        nums_with_index = [(num, i) for i, num in enumerate(nums)]
        nums_with_index.sort()
        left, right = 0, len(nums) - 1
        
        while left < right:
            current_sum = nums_with_index[left][0] + nums_with_index[right][0]
            
            if current_sum == target:
                return [nums_with_index[left][1], nums_with_index[right][1]]
            elif current_sum < target:
                left += 1
            else:
                right -= 1
                

test = Solution()
# print(test.twoSum([2,7,11,15], 9))
# print(test.func([3,2,4], 6))
print(test.func([0,4,3,0], 0))