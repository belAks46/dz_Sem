def evklid(a, b):
     if b == 0:
         return 1, 0, a
     else:
         d, x, y = evklid(b, a % b)
         return d, y, x - y * (a // b)
m, n = map(int, input().split())
print(evklid(m, n))