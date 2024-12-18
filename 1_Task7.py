
class Vector:

    def __init__(self, x, y, z):
        assert isinstance(x, (int, float))
        assert isinstance(y, (int, float))
        assert isinstance(z, (int, float))
        self.x = x
        self.y = y
        self.z = z


    def __abs__(self):
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5

    def __print__(self):
        return (self.x, self.y, self.z)


    def __add__(self, other):
        assert isinstance(other, Vector)
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)


    def __sub__(self, other):
        assert isinstance(other, Vector)
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)


    def __mul__(self, other):
        if isinstance(other, Vector):
            return self.x * other.x + self.y * other.y + self.z * other.z
        if isinstance(other, float):
            return Vector(self.x * other, self.y * other, self.z * other)
        else:
            raise AssertionError

f = Vector(1, 1, 1)
j = Vector(1, 2, 3)
print(f.__abs__())
a = j.__add__(f)
print(a.__print__())
z = f.__mul__(j)
print(z)