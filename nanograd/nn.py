import random
from nanograd.tensor import Scalar

class Module:
    """Parent class @ as pytorch's nn.Module """

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        raise NotImplementedError


class Neuron(Module):
    def __init__(self, n_input):
        self.w = [Scalar(random.uniform(-1, 1)) for _ in range(n_input)]
        self.b = Scalar(0.0)
        self.activation = "relu"

    def __call__(self, x):
        # performs w*x + b
        a = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.activation == "relu":
            out = a.relu()
        else:
            out = a.tanh()  # make this dynamical
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, n_input, n_output):
        self.neurons = [Neuron(n_input) for _ in range(n_output)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Module):
    def __init__(self, n_input, n_output):
        net_size = [n_input] + n_output
        self.layers = [Layer(net_size[i], net_size[i + 1]) for i in range(len(n_output))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
