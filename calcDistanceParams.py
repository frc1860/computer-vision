from math import atan
from time import time

import numpy as np
from scipy.optimize import minimize, root_scalar

D = [200 - 20, 300 - 20, 400 - 20, 413 - 20, 600 - 20]
Y = [28.5, 97.5, 137.5, 145.5, 194.5]
n = len(D)

initialTime = time()

lastPosition = n - 1
midPosition = int(n / 2)
firstPosition = 0


def initialGuessFunction(a):
    return (
        (Y[firstPosition] - Y[midPosition])
        / (Y[firstPosition] - Y[lastPosition])
        * (atan(a / D[firstPosition]) - atan(a / D[lastPosition]))
        - atan(a / D[firstPosition])
        + atan(a / D[midPosition])
    )


initialSolution = root_scalar(initialGuessFunction, x0=160, bracket=[1, 500])


def squaredErrorFunction(x):
    a = x[0]
    b = x[1]
    c = x[2]
    error = 0
    for i in range(n):
        error += (D[i] - (a / np.tan(b * Y[i] + c))) ** 2
    return error


aMin = initialSolution.root - 50
aMax = initialSolution.root + 50
a = initialSolution.root
b = (atan(a / D[firstPosition]) - atan(a / D[midPosition])) / (
    Y[firstPosition] - Y[midPosition]
)
c = atan(a / D[firstPosition]) - b * Y[firstPosition]

finalSolution = [a, b, c]
finalSolutionError = squaredErrorFunction([a, b, c])
a = aMin
while a < aMax:
    x0 = np.array([a, b, c])
    minimizeSolution = minimize(
        squaredErrorFunction, x0, bounds=[(aMin, aMax), (-1, 0), (0.1, 2)]
    )
    if minimizeSolution.success:
        if minimizeSolution.fun < finalSolutionError:
            finalSolution = minimizeSolution.x
            finalSolutionError = minimizeSolution.fun
        if int(100 * a) % 50 == 0:
            print("\n", round(a - aMin, 2), "/", round(aMax - aMin, 2))
            print("error: ", round(minimizeSolution.fun, 6))
            print("minimum:", round(finalSolutionError, 6))
            print(minimizeSolution.x)
    a += 0.01
print("finalSolution:")
print(finalSolution)
finalTime = time()
print("time: {}".format(finalTime - initialTime))
