import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/mmaks/OneDrive/Рабочий стол/лютый прогер/BTC_data (1).csv')

plt.figure(figsize=(10,10))
plt.plot(df['time'], df['close'], color="purple", label="Акции Курского аккумуляторного завода")
plt.xlabel('Время фиксировать прибыль')
plt.ylabel('Цена в грязных зеленых бумажках')

plt.xticks([])
plt.yticks()

plt.grid(axis='both', linestyle='--')
plt.legend()
plt.savefig('C:/Users/mmaks/OneDrive/Рабочий стол/лютый прогер/grafiki sem4.png')
plt.show()