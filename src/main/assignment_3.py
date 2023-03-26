def function(y, t):
    return t - (y ** 2)

def eulers_method(initial_condition, endpoint_a, endpoint_b, iterations):
    step = (endpoint_b - endpoint_a) / iterations
    t, w = endpoint_a, initial_condition

    for i in range(iterations):
        w = w + (step * (function(w, t)))
        t += step

    return w


if __name__ == "__main__":
    # Task One: use Euler's Method to generate approximation of y(t)
    initial_condition = 1
    endpoint_a, endpoint_b = 0, 2
    iterations = 10
    print(eulers_method(initial_condition, endpoint_a, endpoint_b, iterations))