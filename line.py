from numpy import random

EPOCHS = 40_000
RANDOM_ADJUSTMENTS = 32
STEP_SIZE = 0.0001
VERBOSE = False

def cost_function(outputs, expected_values):
    total_cost = 0

    # evaluate the sum of every
    # output comparing it to the
    # expected value, squaring
    # the difference
    for o, y in zip(outputs, expected_values):
        total_cost += (o - y) * (o - y) # () ** 2 is stupid

    # return the sum divided
    # by the number of outputs
    return total_cost / len(outputs)

def evaluate_outputs(coeff, y_int, inputs):
    # compute the outputs by
    # plugging the inputs in the
    # current line to get the outputs
    return [coeff * x + y_int for x in inputs]

def min_random_adjustment(coeff, y_int, inputs, expected_values):
    # generating random numbers
    # without using the random
    # standard library because it'ss dumb
    random_adj_coeff = [random.normal(0, 1) * STEP_SIZE for _ in range(RANDOM_ADJUSTMENTS)]
    random_adj_y_int = [random.normal(0, 1) * STEP_SIZE for _ in range(RANDOM_ADJUSTMENTS)]

    average_costs = []

    # compute the average cost
    # of every single adjustment
    for idx, (rand_coeff, rand_y_int) in enumerate(zip(random_adj_coeff, random_adj_y_int)):
        # compute the outputs with the adjustments
        # applied to the current values
        outputs_adj = evaluate_outputs(coeff + rand_coeff, y_int + rand_y_int, inputs)

        # compute the current average cost
        average_cost_adj = cost_function(outputs_adj, expected_values)

        average_costs.append((idx, average_cost_adj))

    index = min(average_costs, key=lambda t: t[1])[0]

    # return the best adjustments
    return random_adj_coeff[index], random_adj_y_int[index]

dataset = [(-2, 5), (3, 7), (6, 8.2), (-10, 1.8), (-5, 3.8)]

# extract the input
# and the output vectors
# from the given dataset
inputs = [x for x, _ in dataset]
expected_values = [y for _, y in dataset]

# starting values
random_coeff = 3.6
random_y_int = 5.2

for epoch in range(EPOCHS):
    # compute the best adjustments
    best_adj_coeff, best_adj_y_int = min_random_adjustment(random_coeff, random_y_int, inputs, expected_values)

    # implement the adjustments
    random_coeff += best_adj_coeff
    random_y_int += best_adj_y_int

    # print the current average cost
    if VERBOSE:
        # compute the outputs
        outputs = evaluate_outputs(random_coeff, random_y_int, inputs)

        # compute the average cost
        average_cost = cost_function(outputs, expected_values)

        print(f"Epoch number {epoch + 1}, average cost is: {average_cost:.15f}") # should decrease with each epoch

# print the final result
print(random_coeff, random_y_int)