n, m = [int(i) for i in input().split()]
matrix = []
for i in range(n):
    temp = [0 for num in range(m)]
    matrix.append(temp)
x,y,= 0, 0
k = int(1)
matrix[0][0]=1
while k < m*n:
    if x + 1 < m:
        if matrix[y][x+1] == 0:
            x = x + 1
            k = k + 1
            matrix[y][x] = k
            continue
    if y + 1 < n:
        if matrix[y+1][x] == 0:
            y = y + 1
            k = k + 1
            matrix[y][x] = k
            continue
    if matrix[y][x-1] == 0 and x - 1 >= 0:
        x = x - 1
        k = k + 1
        matrix[y][x] = k
        continue
    while matrix[y-1][x] == 0:
        y = y - 1
        k = k + 1
        matrix[y][x] = k
for row in matrix:
  print(*row)
