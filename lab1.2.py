import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Данные
dates = np.array([15, 14, 13, 10, 9, 8, 7, 6, 3, 2, 1, 28, 27, 24, 23, 22, 21, 20, 17, 16, 15])
values = np.array([1.2060, 1.2158, 1.2181, 1.2028, 1.1925, 1.1842, 1.1827, 1.2021, 1.2040, 1.1941, 1.2031, 1.2020, 1.2062, 1.1944, 1.2013, 1.2044, 1.2115, 1.2038, 1.2035, 1.1993, 1.2026])

# Линейная модель
linear_model = np.polyfit(dates, values, 1)
linear_model_fn = np.poly1d(linear_model)
linear_model_r2 = r2_score(values, linear_model_fn(dates))

# Полиномиальная модель
poly_model = np.polyfit(dates, values, 2)
poly_model_fn = np.poly1d(poly_model)
poly_model_r2 = r2_score(values, poly_model_fn(dates))

# Степенная модель
log_dates = np.log(dates)
log_values = np.log(values)
power_model = np.polyfit(log_dates, log_values, 1)
power_model_fn = lambda x: np.exp(power_model[1]) * x ** power_model[0]
power_model_r2 = r2_score(values, power_model_fn(dates))

# Логарифмическая модель
log_model = np.polyfit(dates, log_values, 1)
log_model_fn = lambda x: np.exp(log_model[1]) * np.log(x) ** log_model[0]
log_model_r2 = r2_score(values, log_model_fn(dates))

# Вывод результатов
print(f"Линейная модель: R^2 = {linear_model_r2}")
print(f"Полиномиальная модель: R^2 = {poly_model_r2}")
print(f"Степенная модель: R^2 = {power_model_r2}")
print(f"Логарифмическая модель: R^2 = {log_model_r2}")

# График
plt.figure(figsize=(10, 6))
plt.scatter(dates, values, color='blue')
plt.plot(dates, linear_model_fn(dates), label='Линейная модель', color='red')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(dates, values, color='blue')
plt.plot(dates, poly_model_fn(dates), label='Полиномиальная модель', color='green')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(dates, values, color='blue')
plt.plot(dates, power_model_fn(dates), label='Степенная модель', color='purple')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(dates, values, color='blue')
plt.plot(dates, log_model_fn(dates), label='Логарифмическая модель', color='orange')
plt.legend()
plt.show()
