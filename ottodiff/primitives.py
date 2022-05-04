
import numpy as np
import scipy.sparse as sp

from ottodiff.exceptions import (
    InputUndefinedException, InvalidOperationException, LayerOrderingException,
    MixedGraphsException, ShapeMismatchException, TypeException
    )
from ottodiff.util import flatten_shape

class tensor():
    def __init__(self, upper_shape, lower_shape, value=None, tensor_type=np.float32):
        self.upper_shape = upper_shape
        self.lower_shape = lower_shape
        self.shape = upper_shape + lower_shape
        self.flattened_upper_dim = flatten_shape(upper_shape)
        self.flattened_lower_dim = flatten_shape(lower_shape)

        self.type = tensor_type
        if value is None:
            self.value = sp.csc_matrix((self.flattened_upper_dim,
                                        self.flattened_lower_dim),
                                       dtype=self.type)
        else:
            if type(value) is np.ndarray:
                self.value = sp.csc_matrix(
                    value.reshape(self.flattened_upper_dim,
                                  self.flattened_lower_dim),
                    dtype=self.type)
            else:
                self.value = value.astype(self.type)

    def array(self):
        return self.value.toarray().reshape(self.shape)

    def transpose(self):
        return tensor(self.lower_shape, self.upper_shape, value=self.value.T)

    # TODO: Add handling for tensor->array multiplication
    def __mul__(self, x):
        if type(x) in [int, float]:
            return tensor(self.upper_shape, self.lower_shape,
                          value=self.value * x, tensor_type=self.type)
        if not np.all(self.lower_shape == x.upper_shape):
            raise ShapeMismatchException
        if self.type != x.type:
            raise TypeException
        value = self.value.dot(x.value)
        return tensor(self.upper_shape, x.lower_shape, value=value,
                      tensor_type=self.type)

    def __rmul__(self, x):
        if type(x) in [int, float]:
            return self * x
        raise InvalidOperationException

    def __add__(self, x):
        # TODO: Add tensor scalar addition
        if type(x) in [int, float]:
            return tensor(self.upper_shape, self.lower_shape,
                          value=self.value + x, tensor_type=self.type)
        if not np.all(self.upper_shape == x.upper_shape):
            raise ShapeMismatchException
        if not np.all(self.lower_shape == x.lower_shape):
            raise ShapeMismatchException
        if not self.type == x.type:
            raise TypeException
        return tensor(self.upper_shape, self.lower_shape,
                      self.value + x.value, tensor_type=self.type)

    def __radd__(self, x):
        if type(x) in [int, float]:
            return self + x
        raise InvalidOperationException


class ComputationGraph():
    def __init__(self, graph_type=np.float32):
        self.type = graph_type
        self.reset()

    def reset(self):
        self.unnamed_nodes = 0
        self.num_nodes = 0
        self.layers = []

    def check_inputs_defined(self):
        if self.layers:
            for node in self.layers[0]:
                node.value()

    def check_graph_consistency(self):
        for layer in self.layers:
            for node in layer:
                node.check_graph()
                node.check_layer()

    def check_all(self):
        self.check_inputs_defined()
        self.check_graph_consistency()

    def print_graph(self):
        print('Graph type: %s' % str(self.type))
        for i, layer in enumerate(self.layers):
            print('Layer %d:' % i)
            for node in layer:
                parent_str = ', '.join([n.name for n in node.parents])
                # if node.value is not None:
                #     value_str = 'value = ' + str(node.value())
                # else:
                #     value_str = ''
                print('    %s: %s <-- (%s), shape = %s, id = %d' % (
                    type(node).__name__, node.name, parent_str, str(node.shape), node.id))

    # Node should have parents defined as nodes within the graph before this is called
    def add_node(self, node):
        # Define node layers
        node.layer = 0
        for parent_node in node.parents:
            node.layer = max(parent_node.layer + 1, node.layer)

        if node.layer > len(self.layers):
            raise LayerOrderingException
        if node.layer == len(self.layers):
            self.layers.append([node])
        else:
            self.layers[node.layer].append(node)
        node.graph = self

        node.id = self.num_nodes
        self.num_nodes += 1

        node.type = self.type

        if node.name is None:
            node.name = 'unnamed_%d' % self.unnamed_nodes
            self.unnamed_nodes += 1

    def forward(self):
        self.check_inputs_defined()
        for layer in self.layers[1:]:
            for node in layer:
                node._forward()

    def backward(self, node):
        # Set default derivatives (0 for all nodes
        # but 1 for the node itself)
        # The derivative da/db is stored as a tensor
        # with upper_shape = a.shape and lower_shape = b.shape
        # This means that da/dc = da/db * db/dc != db/dc * da/db
        # (tensor multiplications do not commute in general)
        if type(node) is list:
            for n in node:
                self.backward(n)
        else:
            for layer in self.layers:
                for graph_node in layer:
                    graph_node.derivatives[node.id] = tensor(node.shape, graph_node.shape)
            node.derivatives[node.id] = tensor(node.shape, node.shape, value=sp.eye(flatten_shape(node.shape)))
            node._backward(node)
            for layer in self.layers[1:node.layer][::-1]:
                for graph_node in layer:
                    graph_node._backward(node)

    def get_mutable_nodes(self):
        if len(self.layers):
            return [x for x in self.layers[0] if not x.fixed]
        return []


# The default graph used for computations unless another is specified
DefaultGraph = ComputationGraph()


# General utils
def set_default_graph_type(graph_type):
    if DefaultGraph.num_nodes:
        print("The default graph is not empty, please reset_default_graph(type) instead")
    else:
        DefaultGraph.type = graph_type


def reset_default_graph(graph_type=None):
    if graph_type is None:
        DefaultGraph.reset()
    else:
        DefaultGraph.reset()
        set_default_graph_type(graph_type)


def forward():
    DefaultGraph.forward()


def backward():
    DefaultGraph.backward()


class GraphNode():
    def __init__(self, name=None, graph=None, parents=[],
                 value=None, shape=None, fixed=False):
        # Prevent name from being an integer
        self.id = None
        self.name = name
        self.shape = shape
        self.val = None
        self.layer = None
        self.parents = []
        self.children = []
        self.derivatives = {}
        self.last_modified = None

        # Unfixed level 0 nodes will be modified by default nn optimizer
        self.fixed = fixed

        self.add_parents(parents)

        if graph is None:
            if len(parents):
                graph = parents[0].graph
            else:
                graph = DefaultGraph
        graph.add_node(self)

        if value is not None:
            self.set_value(value)

    def set_value(self, value):
        if type(value) is np.ndarray:
            self.val = value.astype(self.type)
        else: # Assuming value is float or int
            self.val = np.array([value], dtype=self.type)
        self.shape = self.val.shape

    def add_parents(self, parents):
        self.parents += parents
        for node in self.parents:
            node.children.append(self)

    def check_graph(self):
        for node in self.parents:
            if not node.graph is self.graph:
                raise MixedGraphsException
        for node in self.children:
            if not node.graph is self.graph:
                raise MixedGraphsException

    def check_layer(self):
        if self.layer < 0:
            raise LayerOrderingException
        for node in self.parents:
            if not node.layer < self.layer:
                raise LayerOrderingException
        for node in self.children:
            if not node.layer > self.layer:
                raise LayerOrderingException

    def value(self):
        if self.val is None:
            raise InputUndefinedException
        if self.val.dtype != self.type:
            print(self.name, self.type, self.val.dtype)
            raise TypeException
        return self.val
