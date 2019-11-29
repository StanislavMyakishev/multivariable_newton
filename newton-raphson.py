import numpy as np
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

    return np.array([derivative_x1, derivative_x2])


def analytical_gradient(x1, x2):
    """
    Analytical version of gradient
    :param x1: first parameter
    :param x2: second parameter
    :return: analytical form of the gradient for given function
    """
    derivative_x1 = 2 * (2 * x1 * (x1 ** 2 + x2 - 11) + x1 + x2 ** 2 - 7)
    derivative_x2 = 2 * (x1 ** 2 + 2 * x2 * (x1 + x2 ** 2 - 7) + x2 - 11)

    return np.array([derivative_x1, derivative_x2])


def hesseMatrix(x,y):
    x, y = symbols('x y')
    diff(x)
    F = (x ** 2 + y- 11) ** 2 + (x + y ** 2 - 7) ** 2


if __name__ == '__main__':
    print(hesseMatrix())
