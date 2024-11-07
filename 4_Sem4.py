import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/mmaks/OneDrive/Рабочий стол/лютый прогер/iris_data.csv')


pool = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

combinations = [('SepalLengthCm', 'SepalWidthCm'), ('SepalLengthCm', 'PetalLengthCm'),('SepalLengthCm', 'PetalWidthCm'), ('SepalWidthCm', 'PetalLengthCm'), ('SepalWidthCm', 'PetalWidthCm'),('PetalLengthCm', 'PetalWidthCm')]

for x, y in combinations:
    print(x, y)
    plt.figure(figsize=(10, 10))

#расставляем точки из массивов
    plt.scatter(df[x], df[y], marker="1", color='r', label='Исходные данные')

#коэффициенты МНК
    coeffs = np.polyfit(df[x], df[y], 1)

    poly_eq = np.poly1d(coeffs)

    plt.plot(df[x], poly_eq(df[x]), color='blue', label='МНК')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'{x} от {y}')
    plt.legend()
    plt.grid(axis='both', linestyle='--')
    plt.savefig(f'C:/Users/mmaks/OneDrive/Рабочий стол/лютый прогер/grafiki sem4/{x}_and_{y}.png')
    plt.close()