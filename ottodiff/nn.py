import numpy as np
import scipy.sparse as sp

from ottodiff.exceptions import ShapeMismatchException
from ottodiff.operations import ReLuNode, ComputeNode, ElementWiseNode
from ottodiff.primitives import tensor

# Data prep.
def normalize(data, means=None, stds=None):
    if means is None:
        means = np.mean(data, axis=0)
    if stds is None:
        stds = np.std(data, axis=0)

    normed_data = (data - means) / stds

    return normed_data, (means, stds)

# Layers

# Superclass for a nn layer
# This acts as a wrapper around a set of nodes/operations.
# It allows you to set high level parameters of the
# nodes within without actually instantiating them and
# Adding them to the graph. Nodes are instantiated when the
# layer is called (pytorch style).

class InputLayer():
    def __init__(self, in_shape, graph=None):
        self.output = ComputeNode(value=np.zeros(in_shape), graph=graph)

        # Should not be modified during optimization
        self.output.fixed = True
    def feed(self, value):
        self.output.set_value(value)


class LinearLayer():
    def __init__(self, in_dim, out_dim, activation=ReLuNode):#, initialization='rand', use_bias=True):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation
        # weights, biases
        self.weights = ComputeNode(value=np.random.normal(0, 1, (in_dim, out_dim)))
        self.biases = ComputeNode(value=np.random.normal(0, 1, out_dim))

        self.output = None

    def __call__(self, parent):
        if parent.shape[1] != self.in_dim:
            raise ShapeMismatchException
        self.output = self.activation(
            parent.dot(self.weights, [1], [0]) + self.biases)
        return self.output


# Loss functions

class CrossEntropyLossNode(ElementWiseNode):
    def __init__(self, p, y, name=None):
        parents = [p, y]
        super(CrossEntropyLossNode, self).__init__(parents=parents, name=name)

    def _build_derivative_tensors_dynamic(self):
        p = self.parents[0].value().ravel()
        y = self.parents[1].value().ravel()
        p1_derivative = -(y / p - (1 - y) / (1 - p))
        p2_derivative = -(np.log(p) - np.log(1 - p))
        self.derivative_tensors = [tensor(self.shape, self.shape, value=sp.diags(p1_derivative), tensor_type=self.type),
                                   tensor(self.shape, self.shape, value=sp.diags(p2_derivative), tensor_type=self.type)]

    def _forward(self):
        self.set_modified_time()
        p, y = self.parents
        self.val = -(y.value() * np.log(p.value()) + (1 - y.value()) * np.log(1 - p.value()))