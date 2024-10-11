import numpy as np
x = np.array([3,6,7,8,5,6,4])
y = np.array([7,3,9,2,6,4,7])
a = (np.mean(x*y) - np.mean(x)*np.mean(y)) / (np.mean(x**2) * np.mean(x)**2)
b = np.mean(y) - a*np.mean(x)
std_a = 1/np.sqrt(len(x)) * np.sqrt((np.mean(y**2)-np.mean(y)**2)/((np.mean(x**2)-np.mean(x)**2))-a**2)
str_b = std_a* np.sqrt(np.mean(x**2)-np.mean(x)**2)
print(std_a, str_b)