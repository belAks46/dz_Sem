def triangle(h, depth = 1, symbol = '.'):
    if h % 2 != 0 and depth == h//2 + 1:
        print(symbol*depth)
        return
    if h % 2 == 0 and depth == h//2:
        print(symbol*depth)
        print(symbol * depth)
        return
     print(symbol*depth)

