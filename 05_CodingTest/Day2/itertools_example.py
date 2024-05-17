from itertools import combinations, permutations

arr = [1, 2, 3, 4, 5]

comb_result = list(combinations(arr, 3)) ## 5 combination 3
print(f"5 combination 3 : {comb_result}")

perm_result = list(permutations(arr, 3)) ## 5 permutation 3
print(f"5 permutation 3 : {perm_result}")