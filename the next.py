{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "OlfvcEEPR1lp"
   },
   "source": [
    "# Работа 1.1.7\n",
    "# Экспериментальное изучение равноускоренного движения (опыт Галилея)\n",
    "Для запуска блока кода: выделите блок мышкой и нажмите $\\blacktriangleright$ \"Запуск\"  в верхней панели либо *Shift + Enter*\n",
    "### 1.1. Импорт библиотек и установка параметров\n",
    "При необходимости можно изменить номер порта (`port`) или длительность измерения (`measure_time`)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null ,
   "metadata": {
    "id": "EDShcKtXR1lr",
    "slideshow": {
     "slide_type": ""
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import serial # для работы с COM-портом\n",
    "import time\n",
    "import matplotlib.pyplot as plt # построение графиков\n",
    "import numpy as np # работа с массивами\n",
    "import pandas as pd\n",
    "import math\n",
    "from scipy.optimize import curve_fit # подбор параметров наилучшей аппроксимации \n",
    "\n",
    "# Номер порта (если устройство не определяется, попробуйте другие порты в диапазоне COM3 - COM7)\n",
    "port = 'COM3'\n",
    "\n",
    "# время измерения (секунды, только целое, по умолчанию 4)\n",
    "measure_time = 4\n",
    "\n",
    "# технические параметры (НЕ ИЗМЕНЯТЬ)\n",
    "timeout = 1.0 # максимальное время ожидания\n",
    "MAX_VALUES = (measure_time + 1) * 2000\n",
    "MAX_FAIL_COUNT = 3\n",
    "baudrate = 250000 # скорость связи с контроллером\n",
    "#Команды контроллера\n",
    "READY = b'Ready\\r\\n'\n",
    "EXIT = b'Exit\\r\\n'"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1.2 Параметры опыта\n",
    "Внесите в массив ниже координаты катушек на вашей установке (в метрах). Первая катушка *должна* иметь координату (0.0)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Координаты катушек (в метрах)\n",
    "X = np.array([\n",
    "    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9\n",
    "])\n",
    "\n",
    "Ncoils = X.size  # число катушек\n",
    "\n",
    "print(f\"Координаты N = {Ncoils} катушек: {X}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Проведение опыта\n",
    "Выберите и запустите блок кода ниже и следуйте появляющимся указаниям. Для повторного проведения опыта блок можно запустить заново."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# подключение к контроллеру по последовательному порту\n",
    "try:\n",
    "    ser = serial.Serial(port, baudrate, timeout=timeout)\n",
    "except:\n",
    "    print(\"Ошибка подключения. Проверьте кабели или попробуйте другое имя порта.\")\n",
    "    raise\n",
    "else:\n",
    "    print(\"Подключение успешно.\")\n",
    "    \n",
    "# инициализация\n",
    "buffer = []\n",
    "line = ser.readline()\n",
    "if line != b'':\n",
    "    print(\"Линия не готова. Очищаем буфер. Если очистка идет слишком долго,\\\n",
    "    переподключите прибор и перезапустите программу\")\n",
    "    while line != b'':\n",
    "        line = ser.readline()\n",
    "        print('.', end='')\n",
    "print(\"Приготовьте груз (магнит) и поместите его в трубу.\")\n",
    "print(\"Как будете готовы к запуску груза, нажмите Enter...\")\n",
    "input()\n",
    "print(\"Запуск через\")\n",
    "print(\"5\", end=' ')\n",
    "time.sleep(1)\n",
    "print(\"4\", end=' ')\n",
    "time.sleep(1)\n",
    "print(\"3\", end=' ')\n",
    "time.sleep(1)\n",
    "print(\"2\", end=' ')\n",
    "try:\n",
    "    ser.write(str(measure_time).encode('ASCII')) # послать на контроллер команду запуска измрений\n",
    "    time.sleep(1)\n",
    "    print(\"1\", end=' ')\n",
    "    while line != READY:\n",
    "        line = ser.readline() # дождаться готовности\n",
    "    time.sleep(0.5) # 0.5 секунд для регистрации \"пустого\" сигнала\n",
    "    print(\"Поехали!\")\n",
    "    FAIL_COUNT = 0\n",
    "    while (line != EXIT) and (len(buffer) < MAX_VALUES):\n",
    "        # считываем данные\n",
    "        buffer.append(line)\n",
    "        line = ser.readline()\n",
    "        if line == b'': # если слишком часто считывается пустая строка, что-то пошло не так\n",
    "            FAIL_COUNT += 1\n",
    "        if FAIL_COUNT > MAX_FAIL_COUNT:\n",
    "            raise\n",
    "    if len(buffer) == 0:\n",
    "        raise\n",
    "except:\n",
    "    print(\"Что-то пошло не так.\\\n",
    "          Поробуйте повторить опыт заново. Если не помогает:\\\n",
    "          переподключите прибор и перезапустите программу.\")\n",
    "else:\n",
    "    print(\"Готово!\")\n",
    "finally:\n",
    "    ser.close()\n",
    "\n",
    "print(\"Собрано %d значений.\" % len(buffer))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Подготовка сырых данных"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "mNdYU1AeR1lt"
   },
   "outputs": [],
   "source": [
    "data_ = []\n",
    "for dot in buffer:\n",
    "    try:\n",
    "        time_voltage = dot.split(b' ')\n",
    "        if len(time_voltage) == 2:\n",
    "            data_.append([int(time_voltage[0]) / 1e6, int(time_voltage[1])])\n",
    "    except ValueError:\n",
    "        print(\"Предупреждение! Некорректные данные: \", dot)\n",
    "        continue\n",
    "d_time = np.array([d[0] for d in data_])\n",
    "d_voltage = np.array([d[1] for d in data_])\n",
    "print(d_time, d_voltage)\n",
    "\n",
    "plt.figure(figsize=(9,6))\n",
    "plt.plot(d_time, d_voltage, color='blue')\n",
    "plt.title('Зависимость величины регистрируемого сигнала от времени')\n",
    "plt.ylabel('$V$, у.е.')\n",
    "plt.xlabel('$t$, с')\n",
    "plt.grid(which='major', color = 'gray', linewidth=0.5)\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**ВНИМАНИЕ!** Убедитесь в корректности графика выходного сигнала на следующем графике. При обнаружении ошибок переделайте опыт (начиная с блока 2)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Первичная обработка\n",
    "### 4.1 Определение уровня нуля и его погрешности \n",
    "(по умолчанию - по первым 200 точкам)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "Ndots = 200 # количество начальных точек\n",
    "\n",
    "avg = np.mean(d_voltage)\n",
    "semi_avg = np.mean(d_voltage[:Ndots]) # среднее по первым Ndots точкам (нулевой уровень)\n",
    "D = np.std(d_voltage[:Ndots]) # стандартное отклонение\n",
    "plt.figure(figsize=(5, 3))\n",
    "plt.plot(d_voltage[:Ndots])\n",
    "print(\"Среднее = %.3f, стд. отклонение = %.3f (%d точек)\" % (semi_avg, D, Ndots))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4.2 Выделение информативного участка\n",
    "По умолчанию участок определяется автоматически по превышению порогового уровня сигнала. \n",
    "\n",
    "Если участок вырезан неверно, попробуйте изменить параметры в начале блока:\n",
    "* измените порог отклонения сигнала `threshold`  (в единицах стандартного отклонения);\n",
    "* Задайте вручную переменные `MANUAL_START` и `MANUAL_STOP` (в секундах).\n",
    "\n",
    "**ВНИМАНИЕ!** Убедитесь, что максимумы и минимумы на графике ниже сменяют друг друга *последовательно* (не перемежаются). Если последовательность нарушена, **переделайте опыт** (возможно, с другим магнитом)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# критерий отличия сигнала от нуля, в единицах станд. отклонения (по умолчанию - 5σ)\n",
    "threshold = 5.0 \n",
    "\n",
    "# Для вырезания участка вручную установите отличными от нуля (в секундах)\n",
    "MANUAL_START = 0\n",
    "MANUAL_STOP = 0\n",
    "\n",
    "idxs = np.where((abs(d_voltage - semi_avg) > threshold * D))\n",
    "start = idxs[0][0] \n",
    "stop = idxs[0][-1] + 1\n",
    "print(f\"Пороговое напряжение: {threshold * D:.3f} V\")\n",
    "print(f\"Автоматически выделен участок [{d_time[start]:.2f} с, {d_time[stop - 1]:.2f} с] (точки [{start}, {stop}])\")\n",
    "\n",
    "if MANUAL_START > 0 and d_time[-1] > MANUAL_START:\n",
    "    start = np.nonzero(d_time > MANUAL_START)[0][0] - 1\n",
    "    print(f\"ВНИМАНИЕ! Начало участка вручную установлено на {d_time[start]:.2f} с (точка {start})\")\n",
    "if MANUAL_STOP > 0 and d_time[-1] > MANUAL_STOP:\n",
    "    stop = np.nonzero(d_time > MANUAL_STOP)[0][0] - 1\n",
    "    print(f\"ВНИМАНИЕ! Конец участка вручную установлен на {d_time[stop - 1]:.2f} с (точка {stop})\")\n",
    "\n",
    "s_time = d_time[start:stop]\n",
    "s_voltage = d_voltage[start:stop] - semi_avg # отклонение напряжения от среднего (\"нулевого\") значения\n",
    "N = stop - start # число точек\n",
    "\n",
    "# график\n",
    "plt.figure(figsize=(9, 6))\n",
    "plt.plot(s_time, s_voltage, 'b-', alpha=0.5)\n",
    "plt.plot(s_time, s_voltage, 'ro', markersize=3)\n",
    "plt.title('Зависимость величины входного сигнала от времени')\n",
    "plt.ylabel('$V$, у.е.')\n",
    "plt.xlabel('$t$, с')\n",
    "plt.minorticks_on()\n",
    "plt.grid(which='major', linestyle='-')\n",
    "plt.grid(which='minor', axis='x', linestyle='--')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Поиск локальных максимумов и минимумов\n",
    "### 5.1. Определение положения максимумов\n",
    "Первый экстремум принимается за начало отсчёта времени. \n",
    "\n",
    "**ВНИМАНИЕ!** По результатам запуска блока проверьте, что экстремумы определны правильно:\n",
    "* значения не пропущены;\n",
    "* нет ложных экстремумов.\n",
    "\n",
    "При обнаружении ошибки попробуйте изменить порог напряжения (`V_threshold`) или ширину \"окна\" (`t_width`) для поиска максимума и запустите блок ещё раз."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "V_threshold = 0.1 * np.max(s_voltage) # порог напряжения -- по умолчанию 10% от абсолютного максимума\n",
    "t_width = 10 # ширина временного \"окна\" для поиска максимума (в точках) - по умолчанию 10\n",
    "\n",
    "highs = []\n",
    "lows = []\n",
    "last_high = -1\n",
    "last_low = -1\n",
    "for i in range(t_width//2, N - t_width//2):\n",
    "    V = s_voltage[i]\n",
    "    if (V >= V_threshold \n",
    "        and np.all(s_voltage[i - t_width//2: i + t_width//2] <= V)\n",
    "        and i - last_high > t_width):\n",
    "            highs.append(s_time[i])\n",
    "            last_high = i\n",
    "    if (V <= -V_threshold \n",
    "        and np.all(s_voltage[i - t_width//2: i + t_width//2] >= V)\n",
    "        and i - last_low > t_width):\n",
    "            lows.append(s_time[i])\n",
    "            last_low = i\n",
    "\n",
    "# начало отсчёта времени -- первая точка\n",
    "highs = np.array(highs - highs[0])\n",
    "lows = np.array(lows - lows[0])\n",
    "\n",
    "print(\"Времена максимумов, с:\", highs)\n",
    "print(\"Времена минимумов, с:\", lows)\n",
    "\n",
    "if len(highs) != Ncoils or len(lows) != Ncoils:\n",
    "    print(\"ОШИБКА! Число пиков не совпадает с числом катушек!\\\n",
    "    Измените критерии для поиска экстремумов или переделайте опыт\")\n",
    "else:\n",
    "    plt.figure(figsize=(9,6))\n",
    "    plt.plot(highs, X, '^', label='Максимумы')\n",
    "    plt.plot(lows, X, 'v', label='Минимумы')\n",
    "    plt.title(\"Экспериментальные данные\")\n",
    "    plt.ylabel(\"Координата катушек $x$, м\")\n",
    "    plt.xlabel(\"Время $t$, с\")\n",
    "    plt.legend()\n",
    "    plt.grid()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Аппроксимация зависимости и определение ускорения\n",
    "Зависимость координаты груза от времени $x(t)$ должна иметь вид $x(t) = v_0 t + a t^2 /2$ (момент $t=0$ соответствует первому пику). Методом наименьших квадратов подберём оптимальные парметры $v_0$ и $a$, минимизирующие отклонение экспериментальных точек от теоретической кривой. Начальные скорости по максимуму и минимуму сигналов будут разными (так как соответствуют разным положениям груза), однако ускорения должны быть одинаковы в пределах погрешности."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "motion_eq = lambda t, v0, a: v0 * t + 0.5 * a * t**2  # уравнение движения\n",
    "\n",
    "coeffs_h, err_h = curve_fit(motion_eq, highs, X) # подбор по максимумам\n",
    "coeffs_l, err_l = curve_fit(motion_eq, lows, X)  # подбор по минимумам\n",
    "\n",
    "print(\"Ускорение по максимумам / по минимумам:\")\n",
    "print(\"a = %.3f +- %.3f / %.3f +- %.3f [м/с^2]\" % (coeffs_h[1], np.sqrt(err_h[1][1]), \n",
    "                                                   coeffs_l[1], np.sqrt(err_l[1][1])))\n",
    "\n",
    "# график\n",
    "plt.figure(figsize = (9, 6))\n",
    "plt.title(\"Аппроксимация закона движения\")\n",
    "t = np.linspace(0., 1.1 * max(highs[-1], lows[-1]), 100)\n",
    "plt.plot(t, motion_eq(t, *coeffs_h), color=\"blue\", \n",
    "         label=\"Аппроксимация %.3f $t$ + %.3f $t^2$/2\" % (coeffs_h[0], coeffs_h[1]))\n",
    "plt.plot(highs, X, '^', label=\"Исходные данные (максимумы)\")\n",
    "plt.plot(t, motion_eq(t, *coeffs_l), color=\"orange\", \n",
    "         label=\"Аппроксимация %.3f $t$ + %.3f $t^2$/2\" % (coeffs_l[0], coeffs_l[1]))\n",
    "plt.plot(lows, X, 'v', label=\"Исходные данные (минимумы)\")\n",
    "plt.legend()\n",
    "plt.ylabel(\"Координата катушки $x$, м\")\n",
    "plt.xlabel(\"Время $t$, с\")\n",
    "plt.grid()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Определение ускорения свободного падения и коэффициента трения\n",
    "### (блок запускается по результатам серии опытов)\n",
    "Введите в соответствующие массивы в блоке ниже все углы наклона трубы и измеренные в опытах средние значения ускорения и их погрешности."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Углы (в градусах)\n",
    "theta = np.array([\n",
    "    20.2 , 22.4, 24.8, 26.9 , 30.2 , 32.4 , 34.7 , 37.5 , 41.5 , 44.5 , 46.4 , 48.4 , 53.0 , 56.4 , 59.9 ,\n",
    "])\n",
    "\n"
    "# Ускорения (м/с^2)\n",
    "a = np.array([\n",
    "    1.030 , 1.477 , 1.931 , 2.110 , 2.653 , 3.014 , 3.349 , 3.887 , 4.521 , 4.933 , 5.313 , 5.502 , 6.095 , 6.704 , 7.133 ,\n",
    "])\n",
    "\n",
    "# Погрешности углов (в градусах)\n",
    "sigma_theta = np.array([\n",
    "    0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, \n",
    "])\n",
    "\n",
    "# Погрешности ускорений (м/с^2)\n",
    "sigma_a = np.array([\n",
    "    0.010, 0.010, 0.016, 0.014, 0.012, 0.013, 0.022, 0.022, 0.021, 0.024, 0.032, 0.028, 0.034, 0.042, 0.042, \n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# перевод углов в радианы\n",
    "theta_rad = theta / 180 * np.pi\n",
    "sigma_theta_rad = sigma_theta / 180 * np.pi\n",
    "\n",
    "# замена переменных для линеаризации графика\n",
    "ap = a / np.cos(theta_rad)\n",
    "tn = np.tan(theta_rad)\n",
    "\n",
    "# пересчёт погрешностей точек\n",
    "sigma_cos = np.abs(np.sin(theta_rad)) * sigma_theta_rad\n",
    "sigma_tan = sigma_theta_rad / np.cos(theta_rad)**2\n",
    "sigma_ap = ap * np.sqrt((sigma_a / a)**2 + (sigma_cos / np.cos(theta_rad))**2)\n",
    "\n",
    "f = lambda x, k, b: k * x + b\n",
    "\n",
    "K, err = curve_fit(f, tn, ap, sigma=sigma_ap)\n",
    "\n",
    "# результат аппроксимации\n",
    "mu = -K[1]/K[0]\n",
    "g = K[0]\n",
    "\n",
    "# погрешности результа\n",
    "sigma_g = np.sqrt(err[0][0])\n",
    "sigma_mu = mu * np.sqrt(err[0][0] / g**2 + err[1][1] / K[1]**2) \n",
    "\n",
    "print(f\"µ = {mu:.3f} +- {sigma_mu:.3f} ; g = {g:.3f} +- {sigma_g:.3f} м^2/с\")\n",
    "\n",
    "plt.figure(figsize = (9, 6))\n",
    "plt.title(\"Аппроксимация зависимости ускорения от угла наклона\")\n",
    "plt.errorbar(tn, ap, xerr=sigma_tan, yerr=sigma_ap, fmt='ko', ms=3, elinewidth=1, capsize=2)\n",
    "plt.xlabel(r\"$\\tan \\theta$\")\n",
    "plt.ylabel(r\"$a\\: /\\: \\cos \\theta$\")\n",
    "plt.xlim(0, 1.1 * np.max(tn))\n",
    "plt.ylim(0, 1.1 * np.max(ap))\n",
    "plt.plot(tn, f(tn, *K), '-')\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "name": "MainNew-Copy1.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.3"
  },
  "vscode": {
   "interpreter": {
    "hash": "c90aa351139c7a0875e8c113ff78b2d584830a5d698a133dc1c25a689aa2b9d7"
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
