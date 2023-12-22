import math
import random
from lazy_value.models import LazyValue

def net_input_fn(inputs: list[LazyValue], weights: list[float], bias: float) -> float:
    return sum((i.evaluate() * w for i, w in zip(inputs, weights))) + bias

def logistic(value: float) -> float:
    return 1 / (1 + math.exp(-value))

def mse(target: float, value: float) -> float:
    return 0.5 * (target - value) ** 2

class Neuron:
    name: str
    inputs: list[LazyValue]
    weights: list[float]
    bias: float
    target_output_lazy: LazyValue
    net_input_lazy: LazyValue
    error_lazy: LazyValue
    error_pd_lazy: LazyValue
    total_pd_lazy: LazyValue
    input_error_lazy: LazyValue

    def __init__(self, inputs: list[LazyValue], weights: list[float], bias: float, target_output: LazyValue, name: str):
        self.name=name
        self.inputs=inputs
        self.weights=weights
        self.bias=bias
        self.target_output_lazy=target_output
        self.net_input_lazy = LazyValue(net_input_fn, self.inputs, self.weights, self.bias)
        self.output_lazy = LazyValue(logistic, self.net_input_lazy)
        self.error_lazy = LazyValue(mse, self.target_output_lazy, self.output_lazy)
        self.error_pd_lazy = LazyValue(lambda target, output: output - target, self.target_output_lazy, self.output_lazy)
        self.total_pd_lazy = LazyValue(lambda output: output * (1 - output), self.output_lazy)
        self.input_error_lazy = LazyValue(lambda error_pd, total_pd: error_pd * total_pd, self.error_pd_lazy, self.total_pd_lazy)

    def net_input(self) -> float:
        return self.net_input_lazy.evaluate()
    
    def output(self) -> float:
        return self.output_lazy.evaluate()
    
    def calculate_error(self) -> float:
        return self.error_lazy.evaluate()
    
    def calculate_error_pd(self) -> float:
        return self.error_pd_lazy.evaluate()
    
    def calculate_total_pd(self) -> float:
        return self.total_pd_lazy.evaluate()
    
    def calculate_delta(self) -> float:
        return self.input_error_lazy.evaluate()
    
    def update_weights(self, delta: float, learning_rate: float) -> None:
        for weight_index in range(0, len(self.weights)):
            error = delta * self.inputs[weight_index].evaluate()
            self.weights[weight_index] -= learning_rate * error

class Network:
    inputs: list[LazyValue]
    layer_dims: list[int]
    layers: list[list[Neuron]]
    learning_rate: float

    def __init__(self, layers: list[int], inputs: list[LazyValue], learning_rate: float):
        # Build neural network connecting layers to each other
        self.inputs = inputs
        self.layer_dims = layers
        self.layers = []
        self.learning_rate = learning_rate
        inputs = self.inputs
        for layer_index, neuron_count in enumerate(layers[1:]):
            layer: list[Neuron] = []
            for neuron_index in range(0, neuron_count):
                layer.append(Neuron(
                    name=f'{layer_index+1}_{neuron_index}',
                    inputs=inputs,
                    weights=[random.random() for _ in range(0, len(inputs))],
                    bias=random.random(),
                    # Not real value
                    target_output=LazyValue(0, evaluated=True)
                ))
            self.layers.append(layer)
            # next layer gets this layer outputs
            inputs = [n.output_lazy for n in layer]

    def feed_forward(self, inputs: list[float]):
        if len(inputs) != self.layer_dims[0]:
            raise Exception(f'Inputs length must be {self.layer_dims[0]}')
        # set input values
        for index, input in enumerate(inputs):
            self.inputs[index].value = input
        # calculate from back
        return [neuron.output() for neuron in self.layers[-1]]
    
    def train(self, inputs: list[float], outputs: list[float]):
        self.feed_forward(inputs)
        expected_outputs = outputs

        # output layer deltas
        output_layer = self.layers[-1]
        output_deltas = [0.] * len(output_layer)
        for neuron_index, neuron in enumerate(output_layer):
            neuron.target_output_lazy.value = expected_outputs[neuron_index]
            output_deltas[neuron_index] = neuron.calculate_delta()

        # hidden layer deltas
        hidden_layer = self.layers[-2]
        hidden_deltas = [0.] * len(hidden_layer)
        for neuron_index, neuron in enumerate(hidden_layer):
            # sum deltas from output with this weight
            total_delta = sum([
                output_deltas[oi] * o.weights[neuron_index]
                for oi, o in enumerate(output_layer)
            ])
            hidden_deltas[neuron_index] = total_delta * neuron.calculate_total_pd()

        deltas = [
            hidden_deltas,
            output_deltas
        ]

        # update weights
        for layer_index, layer in list(enumerate(self.layers))[::-1]:
            for neuron_index, neuron in enumerate(layer):
                neuron.update_weights(deltas[layer_index][neuron_index], self.learning_rate)

    def calculate_total_error(self, training_sets: list[tuple[list[float], list[float]]]):
        total_error = 0
        for training_inputs, training_outputs in training_sets:
            self.feed_forward(training_inputs)
            for neuron_index, output in enumerate(training_outputs):
                neuron = self.layers[-1][neuron_index]
                neuron.target_output_lazy.assign(output)
                total_error += neuron.calculate_error()
        return total_error


inputs = [0.05, 0.1]
expected_outputs = [0.01, 0.99]
nn = Network(
    layers=[2, 2, 2],
    inputs=[LazyValue(i) for i in inputs],
    learning_rate=0.5
)
for i in range(10000):
    nn.train(inputs, expected_outputs)
    print(i, round(nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.99]]]), 9))


