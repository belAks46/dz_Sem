a = input()
b = True
c = a
ar = []

for c in a:
    ar.append(c)


for m in range(len(a)):
    if ar[m] == "E":
        ar[m] = "3"
    elif ar[m] == '3':
        ar[m] = 'E'
    elif ar[m] == 'J':
        ar[m] = 'L'
    elif ar[m] == 'L':
        ar[m] = 'J'
    elif ar[m] == 'S':
        ar[m] = '2'
    elif ar[m] == '2':
        ar[m] = 'S'
    elif ar[m] == 'Z':
        ar[m] = '5'
    elif ar[m] == '5':
        ar[m] = 'Z'
mirr = ''.join(ar)[::-1]

for c in 'QDRFCGNBKP':
    if c in c:
        b = False
if b == False and c != c[::-1]:
    print(f'{c} is not a palindrome.')
elif c == c[::-1] and c == mirr and b == True:
    print(f'{c} is a mirrored palindrome.')
elif c != c[::-1] and c == mirr:
    print(f'{c} is a mirrored string.')
elif (c == c[::-1] and c != mirr) or (c == c[::-1] and c == mirr and b == False):
    print(f'{c} is a regular palindrome')