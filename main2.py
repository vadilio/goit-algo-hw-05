# Домашне завдання 5.2
def bin_search_up_bnd(arr, target):
    left, right = 0, len(arr) - 1
    iterations = 0
    upper_bound = None

    while left <= right:
        iterations += 1
        mid = (left + right) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            upper_bound = arr[mid]
            right = mid - 1

    return (iterations, upper_bound)


# Тест:
arr = [0.1, 0.5, 1.2, 2.4, 3.6, 4.8, 5.0, 5.5, 6.7]
target = 3.0
result = bin_search_up_bnd(arr, target)
print("Результат пошуку:", result)
