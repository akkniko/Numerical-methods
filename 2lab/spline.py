import numpy as np
import matplotlib.pyplot as plt

# Исходная функция
def f(x):
    return x + abs(100*x)
    #return  x*x- np.sin(10*x)
# Генерация равномерной сетки
def uniform_nodes(a, b, n):
    return np.linspace(a, b, n)

# Генерация чебышевской сетки
def chebyshev_nodes(a, b, n):
    nodes = []
    for i in range(n):
        x = np.cos((2 * i + 1) * np.pi / (2 * n))
        x = 0.5 * (a + b) + 0.5 * (b - a) * x
        nodes.append(x)
    return np.array(sorted(nodes))

# Решение трёхдиагональной системы методом прогонки с обработкой случая n==1
def tridiagonal_solve(a, b, c, d):
    n = len(d)
    # Если система состоит из одного уравнения, сразу возвращаем решение
    if n == 1:
        return np.array([d[0] / b[0]])
    
    c_prime = np.zeros(n-1)
    d_prime = np.zeros(n)
    x = np.zeros(n)
    
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    
    for i in range(1, n-1):
        denom = b[i] - a[i-1] * c_prime[i-1]
        c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i-1] * d_prime[i-1]) / denom
    
    d_prime[n-1] = (d[n-1] - a[n-2] * d_prime[n-2]) / (b[n-1] - a[n-2] * c_prime[n-2])
    x[n-1] = d_prime[n-1]
    
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    
    return x

# Построение кубического сплайна вручную
def cubic_spline(x_nodes, y_nodes):
    n = len(x_nodes) - 1  # число интервалов
    h = np.diff(x_nodes)  # шаги между узлами
    
    coeffs = []
    if n == 1:  # только один интервал
        a_i = y_nodes[0]
        h_i = h[0]
        b_i = (y_nodes[1] - y_nodes[0]) / h_i
        c_i = 0  # S''(x_0)=S''(x_n)=0
        d_i = 0
        coeffs.append((a_i, b_i, c_i, d_i))
    else:
        # Коэффициенты трёхдиагональной системы для второй производной
        A = h[:-1] / 6       # поддиагональ
        B = (h[:-1] + h[1:]) / 3  # главная диагональ
        C = h[1:] / 6        # наддиагональ
        D = np.diff(np.diff(y_nodes) / h)  # правая часть
        
        # Граничные условия: S''(x_0)=S''(x_n)=0 (естественный сплайн)
        m = np.zeros(n + 1)
        if n > 1:
            m[1:-1] = tridiagonal_solve(A, B, C, D)
        
        # Вычисление коэффициентов для каждого интервала
        for i in range(n):
            a_i = y_nodes[i]
            b_i = (y_nodes[i+1] - y_nodes[i]) / h[i] - h[i]*(m[i+1] + 2*m[i])/6
            c_i = m[i] / 2
            d_i = (m[i+1] - m[i]) / (6 * h[i])
            coeffs.append((a_i, b_i, c_i, d_i))
    
    return coeffs, h

# Оценка значения сплайна в точке x
def evaluate_spline(x, x_nodes, coeffs, h):
    n = len(x_nodes) - 1
    for i in range(n):
        if x_nodes[i] <= x <= x_nodes[i+1]:
            dx = x - x_nodes[i]
            a_i, b_i, c_i, d_i = coeffs[i]
            return a_i + b_i * dx + c_i * dx**2 + d_i * dx**3
    # Если x вне интервала, возвращаем ближайшее значение
    if x < x_nodes[0]:
        dx = x - x_nodes[0]
        a_i, b_i, c_i, d_i = coeffs[0]
        return a_i + b_i * dx + c_i * dx**2 + d_i * dx**3
    else:
        dx = x - x_nodes[-2]
        a_i, b_i, c_i, d_i = coeffs[-1]
        return a_i + b_i * dx + c_i * dx**2 + d_i * dx**3

# Вычисление максимальной ошибки
def max_error(n, nodes_func):
    a, b = -0.5, 0.5
    nodes = nodes_func(a, b, n)
    f_values = [f(x) for x in nodes]
    coeffs, h = cubic_spline(nodes, f_values)
    
    mid_points = [(nodes[i] + nodes[i+1]) / 2 for i in range(len(nodes)-1)]
    if not mid_points:
        return 0.0
    
    errors = []
    for x in mid_points:
        s_x = evaluate_spline(x, nodes, coeffs, h)
        errors.append(abs(f(x) - s_x))
    return max(errors)

# Параметры
a, b, n = 0, 100, 10

# Узлы и значения для равномерной сетки
uniform_x = uniform_nodes(a, b, n)
f_uniform = [f(x) for x in uniform_x]

# Узлы и значения для чебышевской сетки
cheb_x = chebyshev_nodes(a, b, n)
f_cheb = [f(x) for x in cheb_x]

# Построение сплайнов
coeffs_uniform, h_uniform = cubic_spline(uniform_x, f_uniform)
coeffs_cheb, h_cheb = cubic_spline(cheb_x, f_cheb)

# Точки для графика
x_plot = np.linspace(a, b, 50)
f_plot = [f(x) for x in x_plot]
S_uniform = [evaluate_spline(x, uniform_x, coeffs_uniform, h_uniform) for x in x_plot]
S_cheb = [evaluate_spline(x, cheb_x, coeffs_cheb, h_cheb) for x in x_plot]

# Графики
plt.figure(figsize=(15, 6))

# График интерполяций
plt.subplot(1, 2, 1)
plt.plot(x_plot, f_plot, label='Исходная функция f(x)', color='black', linewidth=2)
plt.plot(x_plot, S_uniform, '--', label='Чебышевская сетка', color='purple')
plt.plot(x_plot, S_cheb, '-.', label='Чебышевская сеткаРавномерная сетка', color='cyan')
plt.scatter(uniform_x, f_uniform, color='purple', marker='o', s=50, label='Равномерные узлы', zorder=3)
plt.scatter(cheb_x, f_cheb, color='cyan', marker='s', s=100, linewidth=2, label='Чебышевские узлы', zorder=3)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.title('Интерполяция кубическими сплайнами, n=3')
plt.grid(True)

# График ошибки
ns = range(2, 50)
max_errors_uniform = [max_error(n, uniform_nodes) for n in ns]
max_errors_cheb = [max_error(n, chebyshev_nodes) for n in ns]

plt.subplot(1, 2, 2)
plt.semilogy(ns,max_errors_cheb , 'o-', label='Равномерная', color='blue')
plt.semilogy(ns, max_errors_uniform, 'bx-', label='Чебышевская', color='red')
plt.xlabel('Число узлов')
plt.ylabel('Максимальная ошибка')
plt.legend()
plt.title('Максимальная ошибка интерполяции от размера сетки')
plt.grid(True, which='both', linestyle='--')

plt.show()
