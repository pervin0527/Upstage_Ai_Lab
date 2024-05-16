def solution(nums, target):
    n = len(nums)

    def recur(start, answer):
        if len(answer) == 2:
            if nums[answer[0]] + nums[answer[1]] == target:
                return answer
            return
        
        for i in range(start, n):
            answer.append(i)
            result = recur(i + 1, answer)
            if result:
                return result
            answer.pop()

    result = recur(start=0, answer=[])

    return result

print(solution([4, 9, 7, 5, 1], 14))