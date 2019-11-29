import numpy as np
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

    return np.array([round(derivative_x1, 10), round(derivative_x2, 10)])


def analytical_gradient(x1, x2):
    """
    Analytical version of gradient
    :param x1: first parameter
    :param x2: second parameter
    :return: analytical form of the gradient for given function
    """
    derivative_x1 = 4 * x1 * (x1 ** 2 + x2 - 11) + 2 * x1 + 2 * x2 ** 2 - 14
    derivative_x2 = 2 * x1 ** 2 + 4 * x2 * (x1 + x2 ** 2 - 7) + 2 * x2 - 22

    return np.array([derivative_x1, derivative_x2])


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
    counter = 0
    while norm(grad) > eps:
        counter += 1
        curVals = np.subtract(curVals, np.dot(hesse, grad))
        grad = gradient(curVals[0], curVals[1])
        hesse = hesseMatrix(curVals[0], curVals[1])
    print(counter)
    print(curVals)
    return f(curVals)


def reverseHesseMatrix(x1, y1):
    x, y = symbols('x y')
    F = Pow(Pow(x, 2) + y - 11, 2) + Pow(x + Pow(y, 2) - 7, 2)
    return np.power(np.array([[diff(F, x, x).subs([(x, x1), (y, y1)]), diff(F, x, y).subs([(x, x1), (y, y1)])],
                              [diff(F, x, y).subs([(x, x1), (y, y1)]), diff(F, y, y).subs([(x, x1), (y, y1)])]]), -1)


if __name__ == '__main__':
    print(newtonRaphson(f, [4, 3], numerical_gradient, reverseHesseMatrix, 0.01))
