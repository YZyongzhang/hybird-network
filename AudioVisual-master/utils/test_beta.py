from sympy import *
import matplotlib.pyplot as plt

a = 3
b = 8
u = symbols('u')
y = integrate((u**(a - 1)) * ((1 - u)**(b - 1)), (u, 0, 1))

x_l = []
y_l = []
for update in range(40000):
    p = update / 40000
    x_l.append(update)
    y_l.append((p**(a - 1)) * ((1 - p)**(b - 1)))
plt.plot(x_l, y_l)
plt.show()