def findSum(A):
    array = A
    total = len(array)
    if total == 0:
        return total
    else:
        return array[0] + findSum(array[1: ])


if __name__=='__main__':
    print(findSum([3,5,9,2,1]))