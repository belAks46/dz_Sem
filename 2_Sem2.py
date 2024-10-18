a = input().split()
b = int(a[0])
z = a[1]
j = 0
k = b
otv = ''

for i in range(len(z) // b):
    eshe = z[j:k]
    otv = otv + eshe[::-1]
    j += b
    k += b

print(otv)