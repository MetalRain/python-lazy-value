import random
from lazy_value.models import LazyValue

ERROR_EPSILON = 1e-6
LEARNING_RATE = 1e-4
REPETITIONS = 100
HIDDEN_LAYER_1 = 12
HIDDEN_LAYER_2 = 6
SETS = 1
DEBUG = True

def relu(value):
    return value if value > 0 else 0.0

def relu_derivative(value):
    return 1.0 if value >= 0 else 0.0

def arg_sum(*args):
    total = args[0]
    for a in args[1:]:
        total += a
    return total

def random_bias():
    return -1.0 + random.random() * 2.0

def random_01():
    return (random.random() * 100.0) % 1.0

def clamp(value: float):
    return max(-1.0, min(1.0, value))

def onehot(value: int, bits: int) -> list[float]:
    return [float((value >> bit) & 1) for bit in range(0, bits)]

class Node:
    input_weights: list[LazyValue]
    input_values: list[LazyValue]
    input_sums: list[LazyValue]
    input_agg: LazyValue
    final_value: LazyValue

    def __init__(self, name):
        self.name = name
        self.bias = random_bias()
        self.agg_fn = arg_sum
        self.lim_fn = relu
        self.lim_dfn = relu_derivative

    def __str__(self):
        return self.name

    def feed_from_nodes(self, nodes):
        # Build lazy function that evaluates previous layers
        self.input_nodes = nodes
        self.input_weights = []
        self.input_values = []
        self.input_sums = []
        for node in nodes:
            # Each node will have:
            # random weight
            self.input_weights.append(LazyValue(random_bias()))
            # value
            self.input_values.append(node.value())
            # lazy sum
            self.input_sums.append(LazyValue(lambda v, w: v*w, self.input_values[-1], self.input_weights[-1]))

        # Feed this into value calculation
        self.feed_from_values([v for v in self.input_sums])
        if DEBUG:
            print(f'{self}, bias {self.bias} feeds from {", ".join([f"{n}:{self.input_weights[i].evaluate()}" for i, n in enumerate(nodes)])}')
        return self
    
    def feed_from_values(self, values):
        # Build lazy function that evaluates result from values
        agg_args = [self.agg_fn, *values]
        self.input_agg = LazyValue(*agg_args)
        self.final_value = LazyValue(lambda v, b: self.lim_fn(v+b), self.input_agg, self.bias)
        return self
    
    def value(self) -> LazyValue:
        # Value = sum(inputs) * weight + bias
        return self.final_value

    def backprop(self, output: float, error: float):
        if abs(error) > ERROR_EPSILON:
            self.final_value.invalidate()
            input = self.input_agg.evaluate()
            self.input_agg.invalidate()
            # Update bias
            # error = (weight_k * error_j) * transfer_derivative(output)
            self_error = (self.bias * error) * self.lim_dfn(output)
            if abs(self_error) > ERROR_EPSILON: 
                    # weight = weight - learning_rate * error * input
                bias_adjust = - LEARNING_RATE * self_error * 1.0
                if DEBUG:
                    print(f"{self} backprop {error} self error {self_error}, adjusting bias {self.bias} + {bias_adjust}")
                self.bias = self.bias + bias_adjust
            if hasattr(self, 'input_nodes'):
                # Update input weights
                for index, node in enumerate(self.input_nodes):
                    # Invalidate current values
                    self.input_sums[index].invalidate()
                    self.input_values[index].invalidate()

                    node_weight = self.input_weights[index].evaluate()
                    node_value = node.value().evaluate()
                    node_input = node.input_agg.evaluate()
                    # error = (weight_k * error_j) * transfer_derivative(output)
                    node_error = (node_weight * error) * self.lim_dfn(node_value)
                    if abs(node_error) > ERROR_EPSILON:
                        node_weight_adjust = -(LEARNING_RATE * node_error * node_input)
                        if DEBUG:
                            print(f"{self}-{node} error {error} adjusting weight {node_weight} + {node_weight_adjust}")
                        # weight = weight - learning_rate * error * input
                        self.input_weights[index].value = clamp(node_weight + node_weight_adjust)
                        # Propagate error
                        node.backprop(node_value, node_error)

def backpropagation():
    # is input even
    test_sets: list[list[float]] = []

    ONE_VALUE = 1.0
    ZERO_VALUE = 0.0
    
    input_count = 4
    for _ in range(0, SETS):
        a = random.randint(0, 10)
        is_even = ONE_VALUE if a % 2 == 0 else ZERO_VALUE
        inputs: list[float] = onehot(a, 4)
        inputs.extend([is_even])
        test_sets.extend([inputs] * REPETITIONS)
    random.shuffle(test_sets)

    input_values = [LazyValue(random_01()) for _ in range(0, input_count)]

    layer_1 = [
        Node(f'input_{i}').feed_from_values([i_v])
        for i, i_v in enumerate(input_values)
    ]

    layer_2 = [Node(f'hidden_1_{i}').feed_from_nodes(layer_1) for i in range(0, HIDDEN_LAYER_1)]
    layer_3 = [Node(f'hidden_2_{i}').feed_from_nodes(layer_2) for i in range(0, HIDDEN_LAYER_2)]

    result_nodes = [Node('output').feed_from_nodes(layer_3)]
    results = {}

    for test_index, test_set in enumerate(test_sets):
        # feed new values
        for i, v in enumerate(test_set[0:-1]):
            input_values[i].value = v
        
        expectation = test_set[-1]
        
        # evaluate result
        # when positive neuron has higher activation
        # network gives divisibility signal
        positive_result = [node.value().evaluate() for node in result_nodes][0]
        positive_expectation = ONE_VALUE if expectation == ONE_VALUE else ZERO_VALUE
        
        total_error = abs(positive_expectation - positive_result)
        if DEBUG:
            print(f'Testset {test_index} result {positive_result} expectation {positive_expectation}')

        if total_error > ERROR_EPSILON:
            result_nodes[0].backprop(positive_result, positive_expectation - positive_result)
            if DEBUG:
                input("Press Enter to continue...")

        results[test_index] = total_error

    total_error = sum(results.values())
    total_tests = len(results)
    avg_error = total_error / total_tests
    print(f'Total avg error: {avg_error} after {total_tests} tests')


if __name__ == '__main__':
    backpropagation()