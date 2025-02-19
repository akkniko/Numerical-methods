import numpy as np
import matplotlib.pyplot as plt
import math

def f(x):
    #return x**2-np.sin(10*x)
    return 100*abs(x)

def df(x):
    if x > 0:
        return 100
    elif x < 0:
        return -100
    else:
        return 0

def uniform_nodes(a, b, n):
    return np.linspace(a, b, n)

def chebyshev_nodes(a, b, n):
    nodes = []
    for i in range(n):
        x = np.cos((2 * i + 1) * np.pi / (2 * n))
        x = 0.5 * (a + b) + 0.5 * (b - a) * x
        nodes.append(x)
    return np.array(sorted(nodes))

def hermite_interpolant(x, nodes, f_values, df_values):
    n = len(nodes)
    result = 0.0
    for j in range(n):
        xj = nodes[j]
        yj = f_values[j]
        dyj = df_values[j]
        
        L = 1.0
        for k in range(n):
            if k != j:
                L *= (x - nodes[k]) / (xj - nodes[k])
        
        L_prime = 0.0
        for m in range(n):
            if m != j:
                L_prime += 1.0 / (xj - nodes[m])
        
        H_j = (1.0 - 2.0 * (x - xj) * L_prime) * (L ** 2)
        H_tilde_j = (x - xj) * (L ** 2)
        
        result += yj * H_j + dyj * H_tilde_j
    return result

def evaluate_hermite(x_vals, nodes, f_values, df_values):
    return [hermite_interpolant(x, nodes, f_values, df_values) for x in x_vals]

def max_error(n, nodes_func):
    a = -0.5
    b = 0.5
    nodes = nodes_func(a, b, n)
    f_values = [f(x) for x in nodes]
    df_values = [df(x) for x in nodes]
    mid_points = []
    sorted_nodes = sorted(nodes)
    for i in range(len(sorted_nodes)-1):
        mid_points.append((sorted_nodes[i] + sorted_nodes[i+1])/2)
    if not mid_points:
        return 0.0
    errors = []
    for x in mid_points:
        H_x = hermite_interpolant(x, nodes, f_values, df_values)
        errors.append(abs(f(x) - H_x))
    return max(errors)

a, b, n = -0.5, 0.5, 5

# Генерация данных
uniform_x = uniform_nodes(a, b, n)
f_uniform = [f(x) for x in uniform_x]
df_uniform = [df(x) for x in uniform_x]

cheb_x = chebyshev_nodes(a, b, n)
f_cheb = [f(x) for x in cheb_x]
df_cheb = [df(x) for x in cheb_x]

x_plot = np.linspace(a, b, 1000)
f_plot = [f(x) for x in x_plot]
H_uniform = evaluate_hermite(x_plot, uniform_x, f_uniform, df_uniform)
H_cheb = evaluate_hermite(x_plot, cheb_x, f_cheb, df_cheb)

# Построение графиков
plt.figure(figsize=(14, 6))

# График интерполяций и узлов
plt.subplot(1, 2, 1)
plt.plot(x_plot, f_plot, label='Исходная функция f(x)', color='black', linewidth=2)
plt.plot(x_plot, H_uniform, '--', label='Равномерная сетка', color='green')
plt.plot(x_plot, H_cheb, '-.', label='Чебышевская сетка', color='red')
plt.scatter(uniform_x, f_uniform, color='green', marker='o', s=50, label='Равномерные узлы', zorder=3)
plt.scatter(cheb_x, f_cheb, color='red', marker='x', s=100, linewidth=2, label='Чебышевские узлы', zorder=3)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.title('Интерполяция функции, n=5')
plt.grid(True)

# График зависимости ошибки от числа узлов
ns = range(1, 50)
max_errors_uniform = [max_error(n, uniform_nodes) for n in ns]
max_errors_cheb = [max_error(n, chebyshev_nodes) for n in ns]

plt.subplot(1, 2, 2)
plt.semilogy(ns, max_errors_uniform, 'o-', label='Равномерная',color='red')
plt.semilogy(ns, max_errors_cheb, 'bx--', label='Чебышёвская', color='green')
plt.xlabel('Число узлов')
plt.ylabel('Максимальная ошибка')
plt.legend()
plt.title('Максимальная ошибка интерполяции от размера сетки')
plt.grid(True, which='both', linestyle='--')


plt.show()