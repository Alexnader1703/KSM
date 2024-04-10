import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

plt.rcParams["figure.figsize"] = (10,5) # размер графиков

# Массивы входных данных
x = [15, 14, 13, 10, 9, 8, 7, 6, 3, 2, 1, 28, 27, 24, 23, 22, 21, 20, 17, 16, 15]
y = [1.2060, 1.2158, 1.2181, 1.2028, 1.1925, 1.1842, 1.1827, 1.2021, 1.2040, 1.1941, 1.2031, 1.2020, 1.2062, 1.1944, 1.2013, 1.2044, 1.2115, 1.2038, 1.2035, 1.1993, 1.2026]


numpy_x = np.array(x)
numpy_y = np.array(y)

# Линии тренда
# линейный (автоматическое создание)
set_line_by_data = np.polyfit(numpy_x, numpy_y, 1) # полином первой степени
linear_trend = np.poly1d(set_line_by_data) # снижение размерности до одномерного массива
print("{0}x + {1}".format(*set_line_by_data)) # формула


set_polinom_by_data = np.polyfit(numpy_x, numpy_y, 5) # работа с полиномом 2 степени
polinom_trend = np.poly1d(set_polinom_by_data) # Рассчитать значение полинома в точках x

polinom_title = "${0}x^2 + {1}x + {2}$".format(*set_polinom_by_data)
print("${0}x^2 + {1}x + {2}$".format(*set_polinom_by_data)) # формула

# логарифмический
set_log_by_data = np.polyfit(np.log(numpy_x), numpy_y, 1) # работа с полиномом 1 степени + логарифмирование x
log_trend = [set_log_by_data[0]*np.log(x) + set_log_by_data[1] for x in numpy_x] # создание одномерного массива для логарифмического тренда
print("${0}ln(x) + {1}$".format(*set_log_by_data))  # формула

# экспоненциальный
set_exp_by_data = np.polyfit(numpy_x, np.log(numpy_y), 1) # работа с полиномом 1 степени + логарифмирование
exp_trend = [np.exp(set_exp_by_data[1]) * np.exp(set_exp_by_data[0] * x) for x in numpy_x] # создание одномерного массива для экспоненциального тренда
print("${1}e^{0}x$".format(*set_exp_by_data))  # формула

# Расчёт R^2
linear_r2 = r2_score(numpy_y, linear_trend(numpy_x))
polinom_r2 = r2_score(numpy_y, polinom_trend(numpy_x))
log_r2 = r2_score(numpy_y, log_trend)
exp_r2 = r2_score(numpy_y, exp_trend)

# Отображение графиков
plt.figure(figsize=(15, 15)) # размер графика
# 2 графика по горизонтали, 2 по вертикали
plt.subplot(2, 2, 1)

# !!! Текущая ячейка - 1 (левый верхний график)
plt.scatter(numpy_x, numpy_y, label = 'data') # точечный график по x_numpy, y_numpy
plt.plot(numpy_x, linear_trend(numpy_x), linestyle='dashed', color="orange", label = 'linear trend') # линейный тренд
plt.grid(color="gainsboro")
plt.legend(loc='upper right', fontsize=12)
plt.title("Линейный \n$R^2=$" + str(linear_r2) + "\n{0}x + {1}".format(*set_line_by_data))

# !!! Текущая ячейка - 2
plt.subplot(2, 2, 2)
plt.scatter(numpy_x, numpy_y, label = 'data') # точечный график по x_numpy, y_numpy
x = np.linspace(numpy_x.min(), numpy_x.max()) # набор данных для x для большей гладкости графика (50 точек)
plt.plot(x, polinom_trend(x), linestyle='dashed', color="orange", label = 'polinomial trend') # полиномиальный тренд
plt.grid(color="gainsboro")
plt.legend(loc = 'center left', fontsize=12, bbox_to_anchor=(1, 0.5))
plt.title("Полиномиальный \n$R^2=$" + str(polinom_r2) + "\n" + polinom_title)

# !!! Текущая ячейка - 3
plt.subplot(2, 2, 3)
plt.scatter(numpy_x, numpy_y, label = 'data') # точечный график по x_numpy, y_numpy
plt.plot(numpy_x, log_trend, linestyle='dashed', color="orange", label = 'log trend') # логарифмический тренд
plt.grid(color="gainsboro")
plt.legend(loc = 'upper right', fontsize=12)
plt.title("Логарифмический \n$R^2=$" + str(log_r2) + "\n${0}ln(x) + {1}$".format(*set_log_by_data))

# !!! Текущая ячейка - 4
plt.subplot(2, 2, 4)
plt.scatter(numpy_x, numpy_y, label = 'data') # точечный график по x_numpy, y_numpy
plt.plot(numpy_x, exp_trend, linestyle='dashed', color="orange", label = 'exp trend')
plt.grid(color="gainsboro")
plt.legend(loc = 'center left', fontsize=12, bbox_to_anchor=(1, 0.5))
plt.title("Экспоненциальный \n$R^2=$" + str(exp_r2) + "\n{1}e^({0}x)".format(*set_exp_by_data))

fig = plt.gcf()
fig.set_size_inches(15, 15)

plt.show();