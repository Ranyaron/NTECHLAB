"""Найти непрерывный подмассив в массиве, содержащий хотя бы одно число,
который имеет наибольшую сумму."""


def findMaxSubArray(A):

    new_arr = []
    max_sum = 0
    x = 1

    for i in A:
        if x > len(A):
            break
        if i < 0:
            x += 1
            continue

        if i > max_sum:
            max_sum = i
            new_arr = []  # если значение больше максимальной суммы, то обнуляем список
            new_arr.append(i)
        spam = i
        for j in range(x, len(A)):
            # делаем проверку на выгодность прибавления минусового значения
            if A[j] < 0:
                jjj = A[j] - A[j] - A[j]
                if i >= jjj:
                    spam += A[j]
                    new_arr.append(A[j])
                    if spam > max_sum:
                        max_sum = spam
                else:
                    x += 1
                    break
            else:
                spam += A[j]
                if spam > max_sum:
                        max_sum = spam
                        new_arr.append(A[j])
        else:
            x += 1

    # если в конце списка есть значения с минусом, то убираем их
    for k in new_arr[::-1]:
        if k < 0:
            new_arr.remove(k)
        else:
            break

    return new_arr, max_sum


arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]  # [4, -1, 2, 1] = 6
# arr = [-1, 2, 3, -9]  # [2, 3] = 5
# arr = [2, -1, 2, 3, -9]  # [2, -1, 2, 3] = 6
# arr = [-1, 2, 3, -9, 11]  # [11] = 11
# arr = [100, -9, 2, -3, 5]  # [100] = 100
# arr = [1, 2, 3]  # [1, 2, 3] = 6
# arr = [-1, -2, -3]  # [] = 0

n_a, m_s = findMaxSubArray(arr)
print(f"Непрерывный подмассив в массиве: {n_a}, его максимальная сумма {m_s}")
