def fastpow(x,n):
    if n == 1:
        return x
    if n % 2 == 0:
        return fastpow(x,n//2)**2
    else:
        return fastpow(x, n-1)*x
print(fastpow(2,10))