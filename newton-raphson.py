import numpy as np
import pandas as pd
from numpy.linalg import inv
from numpy.linalg import norm
from sympy import *


def f(x):
    """
    Represents the f(x...) - multivariable function
    :param x: x parameter
    :return: returns the function value
    """
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def numerical_gradient(x1, x2, dx=1e-6):
    """
    Numerical version of gradient
    :param x1: first parameter
    :param x2: second parameter
    :param dx: dx - function increment
    :return: numerical form of the gradient for given function
    """
    derivative_x1 = (f([x1 + dx, x2]) - f([x1 - dx, x2])) / (2 * dx)
    derivative_x2 = (f([x1, x2 + dx]) - f([x1, x2 - dx])) / (2 * dx)

    return np.array([round(derivative_x1, 20), round(derivative_x2, 20)])


def analytical_gradient(x1, x2):
    """
    Analytical version of gradient
    :param x1: first parameter
    :param x2: second parameter
    :return: analytical form of the gradient for given function
    """
    derivative_x1 = 4 * x1 * (x1 ** 2 + x2 - 11) + 2 * x1 + 2 * x2 ** 2 - 14
    derivative_x2 = 2 * x1 ** 2 + 4 * x2 * (x1 + x2 ** 2 - 7) + 2 * x2 - 22

    return np.array([round(derivative_x1, 20), round(derivative_x2, 20)])


def newtonRaphson(f, x, gradient, hesseMatrix, eps):
    """
    :param f:
    :param x:
    :param grad:
    :param hesse:
    :param eps:
    :return:
    """
    grad = gradient(x[0], x[1])
    curVals = np.array([x[0], x[1]])
    hesse = hesseMatrix(x[0], x[1])
    cnt = 1
    while norm(grad) > eps:
        cnt += 1
        curVals = np.subtract(curVals, np.dot(hesse, grad))
        grad = gradient(curVals[0], curVals[1])
        hesse = hesseMatrix(curVals[0], curVals[1])
    return norm(grad), f(curVals), cnt, curVals


def reverseHesseMatrix(x1, y1):
    x, y = symbols('x y')
    F = Pow(Pow(x, 2) + y - 11, 2) + Pow(x + Pow(y, 2) - 7, 2)
    return inv(np.array([[diff(F, x, x).subs([(x, x1), (y, y1)]), diff(F, x, y).subs([(x, x1), (y, y1)])],
                         [diff(F, x, y).subs([(x, x1), (y, y1)]), diff(F, y, y).subs([(x, x1), (y, y1)])]],
                        dtype='float'))


if __name__ == '__main__':

    tf = open('./output/newton.txt', 'w')

    count = np.array([0])

    def wrap_f(x):
        count[0] += 1
        return f(x)


    def wrap_analytical_gradient(x1, x2):
        count[0] += 2
        return analytical_gradient(x1, x2)


    def wrap_numerical_gradient(x1, x2):
        count[0] += 2
        return numerical_gradient(x1, x2)


    def wrap_reverseHesseMatrix(x1, x2):
        count[0] += 1
        return reverseHesseMatrix(x1, x2)


    iters = []
    counts = []
    accuracy = []
    normGrad = []
    fValue = []
    fMin = []

    tf.write('Начальная точка: (4, 3)\n')
    tf.write('Способ вычисления производной: Аналитический\n')

    for e in [0.1, 0.001, 0.00001]:
        count[0] = 0
        res = newtonRaphson(wrap_f, [4, 3], wrap_analytical_gradient, wrap_reverseHesseMatrix, e)
        counts.append(count[0])
        iters.append(res[2])
        accuracy.append(e)
        normGrad.append(res[0])
        fValue.append(res[1])
        fMin.append(res[3])

    # return norm(grad), f(curVals), cnt
    data = pd.DataFrame({
        'Колличество итераций': iters,
        'Колличество вычислений': counts,
        'Точность': accuracy,
        'Модуль градиента': normGrad,
        'Найденная точка': fValue,
        'Найденное значение': fMin,
    })

    tf.write(data.to_string() + '\n')
    tf.close()
