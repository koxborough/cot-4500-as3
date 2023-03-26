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


if __name__ == "__main__":
    # Task One: use Euler's Method to generate approximation of y(t)
    initial_condition = 1
    endpoint_a, endpoint_b = 0, 2
    iterations = 10
    print(eulers_method(initial_condition, endpoint_a, endpoint_b, iterations))
    print()

    # Task Two: use Runge-Kutta Method (using input from Task One)
    print(runge_cutta_method(initial_condition, endpoint_a, endpoint_b, iterations))