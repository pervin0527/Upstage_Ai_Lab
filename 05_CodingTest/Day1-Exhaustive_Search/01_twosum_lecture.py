nums = [4, 9, 7, 5, 1]

n = len(nums)
for i in range(n):
    for j in range(i+1, n): ## 중복을 허용하지 않기 위해 i+1부터 실행
        print(nums[i], nums[j], 'n+m:', nums[i]+nums[j])

        if nums[i] + nums[j] == 14:
            print(f"i : {i}, j : {j}")