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
        return value if value > 0 else 0

    def arg_sum(*args):
        total = args[0]
        for a in args[1:]:
            total += a
        return total

    def sign(value):
        if value > 0:
            return 1
        if value < 0:
            return -1
        return 0
    
    def random_bias():
        return -1.0 + random.random() * 2.0

    ERROR_EPSILON = 1e-4
    LEARNING_RATE = 1e-1
    BATCH_SIZE = 10000

    class Node:
        def __init__(self):
            self.bias = random_bias()
            self.agg_fn = arg_sum
            self.lim_fn = relu
            # print(f'{self} init')

        def feed_from_nodes(self, nodes):
            # Build lazy function that evaluates previous layers
            # print(f'{self} feeds from nodes {nodes}')
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
            return self
        
        def feed_from_values(self, values):
            # Build lazy function that evaluates result from values
            # print(f'{self} feeds from values {values}')
            agg_args = [self.agg_fn, *values]
            self.input_agg = LazyValue(*agg_args)
            self.final_value = LazyValue(lambda v, b: self.lim_fn(v+b), self.input_agg, self.bias)
            return self
        
        def value(self) -> LazyValue:
            # Value = sum(inputs) * weight + bias
            # print(f'{self} returns value {self.final_value}')
            return self.final_value

        def backprop(self, error):
            # print(f'{self} backprop {error}')
            if abs(error) > ERROR_EPSILON:
                self.input_agg.invalidate()
                self.final_value.invalidate()
                if hasattr(self, 'input_nodes'):
                    # Update weights
                    for input_weight, input_value in zip(self.input_weights, self.input_values):
                        # print(input_weight, input_value)
                        input_weight.value += LEARNING_RATE * random_bias()
                        #print(input_weight, input_value)
                    for input_sum in self.input_sums:
                        input_sum.invalidate()
                    # Update bias?
                    # Propagate error
                    for node in self.input_nodes:
                        # TODO: give better estimate who is responsible
                        node.backprop(error)

    data = [LazyValue(1.0), LazyValue(1.0)]

    layer_1 = [
        Node().feed_from_values([d])
        for d in data
    ]

    layer_2 = [Node().feed_from_nodes(layer_1)] * 20
    layer_3 = [Node().feed_from_nodes(layer_2)] * 10

    result_node = Node().feed_from_nodes(layer_3)

    test_sets = [
        (2.3, 5.6, 7.9),
        (3.2, 2.1, 5.3),
        (1.1, 0.5, 1.6),
        (8.0, -1.0, 7.0)
    ] * 50
    random.shuffle(test_sets)
    results = {}

    for test_set in test_sets:
        data[0].value = test_set[0]
        data[1].value = test_set[1]
        expectation = test_set[2]

        iterations = 0
        while iterations < BATCH_SIZE:
            for d in data:
                d.evaluate()
            
            result = result_node.value().evaluate()
            error = expectation - result
            
            if abs(error) > ERROR_EPSILON:
                #print(f'Set {test_set}: iteration {iterations}, error {error}')
                result_node.backprop(error)
                iterations += 1
            else:
                break
        print(f'Set {test_set}: finished in {iterations}, error {error}')
        results[test_set] = results.get(test_set, []) + [abs(error)]

    total_error = 0
    total_tests = 0
    for test_set, error_values in results.items():
        error_sum = sum(error_values)
        test_count = len(error_values)
        avg_error = error_sum / test_count
        total_error += error_sum
        total_tests += test_count
        print(f'Set {test_set} avg error: {avg_error}')
    print(f'Total avg error: {total_error/total_tests}')
    print(result_node.value())


if __name__ == '__main__':
    backprop()