nums = [4, 9, 7, 5, 1]

n = len(nums)
for i in range(n):
    for j in range(i+1, n):
        for k in range(j+1, n):
            if nums[i] + nums[j] + nums[k] == 17:
                print(f"i : {i}, j : {j}, k : {k}")