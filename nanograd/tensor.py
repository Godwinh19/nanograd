"""
Tensor is scalar
"""
import math


class Scalar:
    def __init__(self, data, _children={}, _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Scalar(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __neg__(self):
        return self * -1

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float"
        out = Scalar(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        th = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Scalar(th, (self,), 'tanh')

        def _backward():
            self.grad += (1 - th ** 2) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        nodes = []
        visited = set()

        # Create a list of all nodes in order to make backward operation
        def topological_sort(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    topological_sort(child)
                nodes.append(v)

        topological_sort(self)

        self.grad = 1.0  # the last node
        for node in reversed(nodes):
            node._backward()
