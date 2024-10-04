#def triangle(h, depth = 1, symbol = '.'):
# if h % 2 != 0 and depth == h//2 + 1:
#      print(symbol*depth)
#      return
#  if h % 2 == 0 and depth == h//2:
#      print(symbol*depth)
#       print(symbol * depth)
#      return

# print(symbol*depth)
# triangle(h, depth = depth + 1)
#  print(symbol*depth)
#   return
def triangle(size, symb):
     for i in range(1, size + 1):
         print(symb * min(i, size - i + 1))

size, symb = input().split()
size = int(size)
triangle(size, str(symb))