from pulp import LpVariable, LpProblem, LpMaximize,LpStatus
import time

# Создаем модель
model = LpProblem(name="max", sense=LpMaximize)

# Инициализируем переменные
x = {i: LpVariable(name=f"x{i}", lowBound=0) for i in range(1, 4)}

# Добавляем ограничения
model += (2*x[1] + 3*x[2] <= 200, "1")
model += (4*x[2] + 6*x[3] <= 120, "2")
model += (5*x[1] + 5*x[2] + 2*x[3] <= 180, "3")
model += (4*x[1] + 7*x[3] <= 138, "4")

# Определяем целевую функцию
model += 25*x[1] + 28*x[2] + 27*x[3]

start = time.time()
# Решаем задачу оптимизации
status = model.solve()
stop = time.time()


# Получаем результаты
print(f"Статус: {model.status}, {LpStatus[model.status]}\n")
print(f"Прибыль: {model.objective.value()}\n")
print("Резульаты:")
for var in x.values():
    print(f"{var.name}= {var.value()}")

print ("\nВремя :")
print(stop - start)
