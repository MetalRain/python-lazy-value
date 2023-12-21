import random
from lazy_value.models import LazyValue

def lazy_aggregate():
    sum = LazyValue(0, evaluated=False)
    rows = [
        dict(a=1),
        dict(a=2),
        dict(a=3),
        dict(a=4),
        dict(a=5),
        dict(a=6),
    ]

    for row in rows:
        row['sum'] = sum
        row['perc'] = LazyValue(lambda a, b: a/b, row['a'], sum)
        sum += row['a']

    # calculates lazy values dependent on sum
    sum.evaluate()

    for row in rows:
        print(row)

def backprop():
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
    
    def clamp(value: float):
        return max(-1.0, min(1.0, value))

    ERROR_EPSILON = 1e-4
    LEARNING_RATE = 1e-2
    REPETITIONS = 1000
    SETS = 1
    DEBUG = False

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
                # Update bias
                # error = (weight_k * error_j) * transfer_derivative(output)
                self_error = (self.bias * error) * self.lim_dfn(output)
                if abs(self_error) > ERROR_EPSILON: 
                     # weight = weight - learning_rate * error * input
                    bias_adjust = - LEARNING_RATE * self_error
                    if DEBUG:
                        print(f"{self} backprop {error} self error {self_error}, adjusting bias {self.bias} + {bias_adjust}")
                    self.bias = clamp(self.bias + bias_adjust)
                if hasattr(self, 'input_nodes'):
                    # Invalidate current values
                    for input_sum in self.input_sums:
                        input_sum.invalidate()
                    # Update input weights
                    self.input_agg.invalidate()
                    for index, node in enumerate(self.input_nodes):
                        node_weight = self.input_weights[index].evaluate()
                        node_value = node.value().evaluate()
                        # error = (weight_k * error_j) * transfer_derivative(output)
                        node_error = (node_weight * error) * self.lim_dfn(node_value)
                        if abs(node_error) > ERROR_EPSILON:
                            node_weight_adjust = -(node_error * LEARNING_RATE * node_value)
                            if DEBUG:
                                print(f"{self}-{node} error {error} adjusting weight {node_weight} + {node_weight_adjust}")
                            # weight = weight - learning_rate * error * input
                            self.input_weights[index].value = clamp(node_weight + node_weight_adjust)
                            # Propagate error
                            node.backprop(node_value, node_error)
                    
                        
    input_values = [LazyValue(1.0), LazyValue(1.0)]

    layer_1 = [
        Node(f'input_{i}').feed_from_values([i_v])
        for i, i_v in enumerate(input_values)
    ]

    layer_2 = [Node(f'hidden_1_{i}').feed_from_nodes(layer_1) for i in range(0, 5)]
    layer_3 = [Node(f'hidden_2_{i}').feed_from_nodes(layer_2) for i in range(0, 2)]

    result_node = Node('output').feed_from_nodes(layer_3)

    test_sets: list[tuple[float, float, float]] = []
    
    for _ in range(0, SETS):
        a = float(random.randint(0, 50) / 100.)
        b = float(random.randint(0, 50) / 100.)
        c = a + b
        test_sets.extend([(a, b, c)] * REPETITIONS)
    random.shuffle(test_sets)
    results = {}

    for test_set in test_sets:
        a, b, expectation = test_set
        input_values[0].value = a
        input_values[1].value = b

        # evaluating triggers propagation
        for i_v in input_values:
            i_v.evaluate()
        
        result = result_node.value().evaluate()
        error = expectation - result
        
        if abs(error) > ERROR_EPSILON:
            print(f'Testset {test_set} result {result} expectation {expectation} error {error}')
            result_node.backprop(result, error)
            if DEBUG:
                input("Press Enter to continue...")
        else:
            print(f'Testset {test_set}  error {error}')
        results[test_set] = results.get(test_set, []) + [abs(error)]

    total_error = 0
    total_tests = 0
    for test_set, error_values in results.items():
        error_sum = sum(error_values)
        test_count = len(error_values)
        avg_error = error_sum / test_count
        total_error += error_sum
        total_tests += test_count
        # print(f'Set {test_set} avg error: {avg_error}')
    print(f'Total avg error: {total_error/total_tests}')


if __name__ == '__main__':
    backprop()