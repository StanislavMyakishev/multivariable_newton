import numpy as np
from sympy import *


def f(x):
    """
    Represents the f(x...) - multivariable function
    :param x: x parameter
    :return: returns the function value
    """
    return (1.5 - x[0] * (1 - x[1])) ** 2 + (2.25 - x[0] * (1 - x[1] ** 2)) ** 2 + (2.625 - x[0] * (1 - x[1] ** 3)) ** 2


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
    derivative_x1 = 2 * (1.5 - x1 * (1 - x2)) * (x2 - 1) + 2 * (2.25 - x1 * (1 - x2 ** 2)) * (x2 ** 2 - 1) \
                    + 2 * (2.625 - x1 * (1 - x2 ** 3)) * (x2 ** 3 - 1)
    derivative_x2 = 2 * (1.5 - x1 * (1 - x2)) * x1 + 2 * (2.25 - x1 * (1 - x2 ** 2)) * 2 * x1 * x2 \
                    + 2 * (2.625 - x1 * (1 - x2 ** 3)) * 3 * (x2 ** 2) * x1

    return np.array([derivative_x1, derivative_x2])


def HesseMatrix(x,y):
    """
        x1, y1 = symbols('x1 y1')
        F = (x1**2 + y1 - 11)**2 + (x1 + y1**2 - 7)**2
        dfdx = diff(lambda x: f(x), x[0])
        dfdy = diff(F, y1)
        dfdxdx = diff(dfdx, x1)
        dfdydy = diff(dfdy, y1)
        dfdxdy = diff(dfdx, y1)
        dfdydx = diff(dfdy, x1)
        a = str(dfdx)
        return np.array([[dfdxdx, dfdxdy], [dfdydx, dfdydy]])
    :param x:
    :param y:
    :return:
    """
    x1, y1 = symbols('x1 y1')
    F = (x1 ** 2 + y1 - 11) ** 2 + (x1 + y1 ** 2 - 7) ** 2
    dfdx = diff(F, x1)
    print(str(dfdx))
    test = "def fq(x, y):\treturn 4*x1*(x1**2 + y1 - 11) + 2*x1 + 2*y1**2 - 14"
    fu = eval(test)
    print(fu(1, 2))

if __name__ == '__main__':
    print(HesseMatrix([1, 2],1))
