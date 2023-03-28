import numpy as np
np.set_printoptions(precision=7, suppress=True, linewidth=100)

def function(t, y):
    return t - (y ** 2)

def eulers_method(initial_condition, endpoint_a, endpoint_b, iterations):
    step = (endpoint_b - endpoint_a) / iterations
    t, w = endpoint_a, initial_condition

    for i in range(iterations):
        w = w + (step * (function(t, w)))
        t += step

    return w

def runge_cutta_method(initial_condition, endpoint_a, endpoint_b, iterations):
    step = (endpoint_b - endpoint_a) / iterations
    t, w = endpoint_a, initial_condition

    for i in range(iterations):
        order_1 = step * function(t, w)
        order_2 = step * function(t + (step / 2), w + (order_1 / 2))
        order_3 = step * function(t + (step / 2), w + (order_2 / 2))
        order_4 = step * function(t + step, w + order_3)

        w = w + ((order_1 + (2 * order_2) + (2 * order_3) + order_4) / 6)
        t += step

    return w

def gauss_jordan(matrix):
    length = len(matrix)
    list = []

    for i in range(length):
        max_row = i
        for j in range(i + 1, length):
            if abs(matrix[j][i]) > abs(matrix[max_row][i]):
                max_row = j

        matrix[[i, max_row]] = matrix[[max_row, i]]

        pivot = matrix[i][i]
        for j in range(i, length + 1):
            matrix[i][j] /= pivot
        
        for j in range(i + 1, length):
            factor = matrix[j, i]
            for k in range(length + 1):
                matrix[j][k] -= (factor * matrix[i][k])

    for i in range(length - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            factor = matrix[j, i]
            for k in range(length + 1):
                matrix[j][k] -= (factor * matrix[i][k])

    for i in range(length):
        list.append(int(matrix[i][length]))
    
    return list

def lu_factorization(matrix):
    length = len(matrix)
    L = np.zeros((length, length))
    U = np.array(matrix)

    for i in range(length):
        L[i][i] = 1

    for i in range(length):
        for j in range(i + 1, length):
            L[j][i] = (U[j][i] / U[i][i])
            U[j] -= (L[j][i] * U[i])

    return np.linalg.det(matrix), L, U

def diagonally_dominant(matrix):
    length = len(matrix)
    for i in range(length):
        sum = -abs(matrix[i][i])
        for j in range(length):
            sum += abs(matrix[i][j])
        
        if (abs(matrix[i][i]) < sum):
            return False
    return True

def positive_definitive(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    for val in eigenvalues:
        if val <= 0:
            return False
    
    return True

if __name__ == "__main__":
    # Task One: use Euler's Method to generate approximation of y(t)
    initial_condition = 1
    endpoint_a, endpoint_b = 0, 2
    iterations = 10
    print(eulers_method(initial_condition, endpoint_a, endpoint_b, iterations))
    print()

    # Task Two: use Runge-Kutta Method (using input from Task One)
    print(runge_cutta_method(initial_condition, endpoint_a, endpoint_b, iterations))
    print()

    # Task Three: use Gaussian elimination and backward substitution to solve linear system
    matrix = np.array([[2., -1., 1., 6.], [1., 3., 1., 0.], [-1., 5., 4., -3.]])
    print(np.array(gauss_jordan(matrix)))
    print()

    # Task Four: implement LU Factorization and find matrix determinant, L matrix, U matrix
    matrix = np.array([[1., 1., 0., 3.], [2., 1., -1., 1.], [3., -1., -1., 2.], [-1., 2., 3., -1.]])
    determinant, L, U = lu_factorization(matrix)
    print(determinant)
    print()
    print(L)
    print()
    print(U)
    print()

    # Task Five: Determine if a matrix is diagonally dominant
    matrix = np.array([[9, 0, 5, 2, 1], [3, 9, 1, 2, 1], [0, 1, 7, 7, 2], [4, 2, 3, 12, 2], [3, 2, 4, 0, 8]])
    print(diagonally_dominant(matrix))
    print()

    # Task Six: Determine if a amtrix is positive definite
    matrix = np.array([[2, 2, 1], [2, 3, 0], [1, 0, 2]])
    print(positive_definitive(matrix))