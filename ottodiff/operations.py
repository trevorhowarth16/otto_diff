from datetime import datetime
import numpy as np
import scipy.sparse as sp

from ottodiff.exceptions import ShapeMismatchException, TooFewInputsException
from ottodiff.primitives import tensor, GraphNode
from ottodiff.util import flatten_shape, broadcast_shapes

# Superclass for all nodes which represent operations
class ComputeNode(GraphNode):
    def derivative(self, node):
        if self.id in node.derivatives:
            return node.derivatives[self.id]
        return None

    def _forward(self):
        raise NotImplementedError


    def _build_derivative_tensors_dynamic(self):
        pass

    # Calculates and stores dnode/dparent (total derivative)
    # for each node that is a parent of self
    def _backward(self, node):
        self._build_derivative_tensors_dynamic()
        # Skip calculation if dnode/dself is 0
        if np.any(self.derivatives[node.id]):
            for parent, derivative in zip(self.parents, self.derivative_tensors):
                parent.derivatives[node.id] += self.derivatives[node.id] * derivative

    def set_modified_time(self):
        self.last_modified = datetime.now()

    def sum(self, sum_indices=None):
        return TensorSumNode(self, indices=sum_indices)

    def reshape(self, shape):
        return ReshapeNode(self, shape)

    def transpose(self, transpose_list):
        return TransposeNode(self, transpose_list)

    def dot(self, other, indices_1, indices_2):
        return TensorDotNode(self, other, indices_1, indices_2)

    def __add__(self, val):
        if type(val) in [int, float]:
            return ScalarAddNode(self, val)
        return AddNode(self, val)

    def __radd__(self, val):
        return self + val

    def __sub__(self, val):
        return self + (-1 * val)

    def __rsub__(self, val):
        return (-1 * self) + val

    def __mul__(self, val):
        if type(val) in [int, float]:
            return ScalarMultiplyNode(self, val)
        return MultiplyNode(self, val)

    def __rmul__(self, val):
        return self * val

    def __div__(self, val):
        return self * (val ** -1)

    def __rdiv__(self, val):
        return (self ** -1) * val

    def __pow__(self, val):
        if type(val) in [int, float]:
            return PowerNode(self, self.type(val))
        else:
            raise NotImplementedError

    def __rpow__(self, val):
        raise NotImplementedError


# Nodes which represent operations

# Broadcasts input from in_shape to out_shape
class BroadcastNode(ComputeNode):
    def __init__(self, in_node, out_shape, name=None):
        self.in_shape = in_node.shape
        self.out_shape = out_shape

        # Checks that shapes are valid
        broadcast_shapes([self.in_shape, self.out_shape])
        super(BroadcastNode, self).__init__(graph=in_node.graph,
                                            parents=[in_node], name=name)
        self._build_derivative_tensors_static()
        self.shape = self.out_shape

    def _forward(self):
        self.set_value(
            np.broadcast_to(self.parents[0].value(),
                            self.out_shape))

    def _build_derivative_tensors_static(self):
        flattened_in_dim = flatten_shape(self.in_shape)
        flattened_out_dim = flatten_shape(self.out_shape)
        col_inds = np.broadcast_to(
            np.arange(flattened_in_dim, dtype=int).reshape(
                self.in_shape), self.out_shape).flatten()
        row_inds = np.arange(flattened_out_dim, dtype=int)
        data = np.ones(flattened_out_dim, dtype=self.type)
        derivative = sp.csc_matrix((data, (row_inds, col_inds)),
                                    shape=(flattened_out_dim, flattened_in_dim))

        self.derivative_tensors = [tensor(self.out_shape, self.in_shape, value=derivative, tensor_type=self.type)]


def broadcast_nodes(nodes):
    out_nodes = []
    shape = broadcast_shapes([x.shape for x in nodes])
    for node in nodes:
        if node.shape == shape:
            out_nodes.append(node)
        else:
            out_nodes.append(BroadcastNode(node, shape))
    return shape, out_nodes

# Base class for any node doing elementwise operations
# This class automatically broadcasts all inputs
class ElementWiseNode(ComputeNode):
    def __init__(self, parents, name=None):
        shape, broadcast_parents = broadcast_nodes(parents)
        super(ElementWiseNode, self).__init__(
            parents=broadcast_parents, name=name,
            shape=shape)


class SumNode(ElementWiseNode):
    def __init__(self, addends, name=None):
        if len(addends) < 2:
            raise TooFewInputsException
        super(SumNode, self).__init__(parents=addends, name=name)
        self._build_derivative_tensors_static()

    def _build_derivative_tensors_static(self):
        flattened_in_dim = flatten_shape(self.shape)
        derivative = sp.eye(flattened_in_dim)

        self.derivative_tensors = [tensor(self.shape, self.shape, value=derivative, tensor_type=self.type) for _ in self.parents]

    def _forward(self):
        self.set_modified_time()
        self.val = np.zeros(self.shape, dtype=self.type)
        for node in self.parents:
            self.val += node.value()


class AddNode(SumNode):
    def __init__(self, parent1, parent2, name=None):
        super(AddNode, self).__init__([parent1, parent2], name=name)


class MultiplyNode(ElementWiseNode):
    def __init__(self, parent1, parent2, name=None):
        parents = [parent1, parent2]
        super(MultiplyNode, self).__init__(parents=parents, name=name)

    def _build_derivative_tensors_dynamic(self):
        p1_derivative = sp.diags(self.parents[1].value().ravel())
        p2_derivative = sp.diags(self.parents[0].value().ravel())

        self.derivative_tensors = [tensor(self.shape, self.shape, value=p1_derivative),
                                   tensor(self.shape, self.shape, value=p2_derivative)]

    def _forward(self):
        self.set_modified_time()
        self.val = self.parents[0].value() * self.parents[1].value()


class TensorDotNode(ComputeNode):
    def __init__(self, parent1, parent2, indices_1, indices_2, name=None):
        # Handling for tensor*tensor -> scalar products
        if len(indices_1) == len(parent1.shape) and len(indices_2) == len(parent2.shape):
            parents = [BroadcastNode(parent1, (1,) + parent1.shape), parent2]
            indices_1 = [x + 1 for x in indices_1]
        else:
            parents = [parent1, parent2]
        super(TensorDotNode, self).__init__(parents=parents, name=name)
        self.indices_1 = indices_1
        self.indices_2 = indices_2
        self.init_shape()

    def init_shape(self):
        if len(self.indices_1) != len(self.indices_2):
            raise ShapeMismatchException
        for dim1, dim2 in zip(self.indices_1, self.indices_2):
            if self.parents[0].shape[dim1] != self.parents[1].shape[dim2]:
                raise ShapeMismatchException
        shape = []
        for i, dim in enumerate(self.parents[0].shape):
            if i not in self.indices_1:
                shape.append(dim)
        for i, dim in enumerate(self.parents[1].shape):
            if i not in self.indices_2:
                shape.append(dim)
        self.shape = tuple(shape)

    def _build_derivative_tensors_dynamic(self):
        shared_shape = tuple([self.parents[0].shape[i] for i in self.indices_1])
        total_shape = self.shape + shared_shape
        output_shape_flat = flatten_shape(self.shape)
        input1_shape_flat = flatten_shape(self.parents[0].shape)
        input2_shape_flat = flatten_shape(self.parents[1].shape)

        row_indices = np.arange(output_shape_flat).reshape(self.shape + tuple(np.ones_like(shared_shape)))
        row_indices = np.broadcast_to(row_indices, total_shape).flatten()

        mesh_grids = np.meshgrid(*[np.arange(x) for x in total_shape], indexing='ij')
        mesh_grids = [x.flatten() for x in mesh_grids]

        # dc1[i] is the dimension of shape1 that the ith dimension of
        # the row_indices tensor corresponds to
        dc1 = []
        for i, _ in enumerate(self.parents[0].shape):
            if i not in self.indices_1:
                dc1.append(i)
        dc1 += ([-1] * (len(self.parents[1].shape) - len(self.indices_2))) + list(self.indices_1)
        ms = [1, 0]
        for i in self.parents[0].shape[::-1][:-1]:
            ms = [ms[0] * i] + ms
        ms = np.array(ms)
        mults = ms[dc1]
        # These are in the column indices for dself/dparent1
        col_indices1 = np.zeros_like(row_indices)
        for mg, mult in zip(mesh_grids, mults):
            col_indices1 += mg * mult

        # dc2[i] is the dimension of shape2 that the ith dimension of
        # the row_indices tensor corresponds to
        dc2 = [-1] * (len(self.parents[0].shape) - len(self.indices_1))
        for i, _ in enumerate(self.parents[1].shape):
            if i not in self.indices_2:
                dc2.append(i)
        dc2 += list(self.indices_2)

        ms = [1, 0]
        for i in self.parents[1].shape[::-1][:-1]:
            ms = [ms[0] * i] + ms
        ms = np.array(ms)
        mults = ms[dc2]
        # These are in the column indices for da/dn1
        col_indices2 = np.zeros_like(row_indices)
        for mg, mult in zip(mesh_grids, mults):
            col_indices2 += mg * mult
        data1 = self.parents[1].value().flatten()[col_indices2]
        derivative1 = sp.csc_matrix((data1, (row_indices, col_indices1)), shape=(output_shape_flat, input1_shape_flat))
        data2 = self.parents[0].value().flatten()[col_indices1]
        derivative2 = sp.csc_matrix((data2, (row_indices, col_indices2)), shape=(output_shape_flat, input2_shape_flat))
        self.derivative_tensors = [tensor(self.shape, self.parents[0].shape, value=derivative1, tensor_type=self.type),
                                   tensor(self.shape, self.parents[1].shape, value=derivative2, tensor_type=self.type)]


    def _forward(self):
        self.val = np.tensordot(
            self.parents[0].value(),
            self.parents[1].value(),
            axes=(self.indices_1, self.indices_2))

class TransposeNode(ComputeNode):
    def __init__(self, parent, transpose_indices, name=None):
        super(TransposeNode, self).__init__(parents=[parent], name=name)
        self.transpose_indices = transpose_indices
        self.shape = tuple([parent.shape[x] for x in transpose_indices])
        self._build_derivative_tensors_static()

    def _build_derivative_tensors_static(self):
        flattened_dim = flatten_shape(self.shape)
        col_inds = np.transpose(
            np.arange(flattened_dim, dtype=int).reshape(
                self.parents[0].shape),
            self.transpose_indices).flatten()
        row_inds = np.arange(flattened_dim, dtype=int)
        data = np.ones(flattened_dim, dtype=np.float32)
        derivative = sp.csc_matrix((data, (row_inds, col_inds)), shape=(flattened_dim, flattened_dim))
        derivative_tensor = tensor(self.shape, self.parents[0].shape, value=derivative, tensor_type=self.type)

        self.derivative_tensors = [derivative_tensor]

    def _forward(self):
        self.val = np.transpose(self.parents[0].value(), self.transpose_indices)

class ReshapeNode(ComputeNode):
    def __init__(self, parent, shape, name=None):
        super(ReshapeNode, self).__init__(parents=[parent], name=name, shape=shape)
        self._build_derivative_tensors_static()

    def _build_derivative_tensors_static(self):
        flattened_dim = flatten_shape(self.shape)
        derivative = sp.eye(flattened_dim)

        derivative_tensor = tensor(self.shape, self.parents[0].shape, value=derivative, tensor_type=self.type)
        self.derivative_tensors = [derivative_tensor]

    def _forward(self):
        self.val = self.parents[0].value().reshape(self.shape)

class TensorSumNode(ComputeNode):
    def __init__(self, parent, indices=None, name=None):
        # Sum over all indices
        if indices is None:
            indices = range(len(parent.shape))
        # Deal with summing down to a scalar
        if len(indices) == len(parent.shape):
            parent = BroadcastNode(parent, (1,) + parent.shape)
            indices = [x + 1 for x in indices]
        shape = []
        for i, dim in enumerate(parent.shape):
            if i not in indices:
                shape.append(dim)
        shape = tuple(shape)
        super(TensorSumNode, self).__init__(parents=[parent], name=name, shape=shape)
        self.sum_indices = tuple(indices)
        self._build_derivative_tensors_static()

    def _build_derivative_tensors_static(self):
        flattened_shape = flatten_shape(self.parents[0].shape)
        flattened_output_shape = flatten_shape(self.shape)
        col_inds = np.arange(flattened_shape, dtype=int)
        row_inds = np.zeros(flattened_shape, dtype=int)
        multiplier = 1
        divisor = 1
        for i, dim in enumerate(self.parents[0].shape[::-1]):
            j = len(self.parents[0].shape) - i - 1
            if j not in self.sum_indices:
                row_inds += multiplier * ((col_inds / divisor).astype(int) % dim)
                multiplier *= dim
            divisor *= dim
        row_inds = row_inds.astype(int)
        data = np.ones(flattened_shape, dtype=np.float32)
        derivative = sp.csr_matrix((data, (row_inds, col_inds)), shape=(flattened_output_shape, flattened_shape))
        derivative_tensor = tensor(self.shape, self.parents[0].shape, value=derivative, tensor_type=self.type)

        self.derivative_tensors = [derivative_tensor]

    def _forward(self):
        self.val = np.sum(self.parents[0].value(), self.sum_indices)

# The superclass for any node which applies
# a function independently to each element in the input array,
# returning an output array of the same shape.
class FunctionNode(ComputeNode):
    def __init__(self, parent, name=None):
        # Sum over all indices
        super(FunctionNode, self).__init__(parents=[parent], name=name, shape=parent.shape)

    def function(self, in_value):
        raise NotImplementedError

    def derivative_function(self, in_value):
        raise NotImplementedError

    def _build_derivative_tensors_dynamic(self):
        flattened_val = self.parents[0].value().ravel()
        flattened_derivative = self.derivative_function(flattened_val)
        derivative = sp.diags(flattened_derivative)

        derivative_tensor = tensor(self.shape, self.shape, value=derivative, tensor_type=self.type)
        self.derivative_tensors = [derivative_tensor]

    def _forward(self):
        self.val = self.function(self.parents[0].value())


class ReLuNode(FunctionNode):
    def __init__(self, parent, leak_constant=0, name=None):
        # Sum over all indices
        super(ReLuNode, self).__init__(parent, name=name)
        self.leak_constant = leak_constant

    def function(self, in_value):
        out_value = in_value.copy()
        out_value[out_value < 0] *= self.leak_constant
        return out_value

    def derivative_function(self, in_value):
        out_value = np.ones_like(in_value, dtype=self.type)
        out_value[out_value < 0] = self.leak_constant
        return out_value


class SigmoidNode(FunctionNode):
    def __init__(self, parent, name=None):
        # Sum over all indices
        super(SigmoidNode, self).__init__(parent, name=name)

    def function(self, in_value):
        out_value = (np.tanh(in_value) + 1.) / 2.
        return out_value

    def derivative_function(self, in_value):
        out_value = 0.5 * (1 - np.tanh(in_value) ** 2)

        return out_value


class PowerNode(FunctionNode):
    def __init__(self, parent, power, name=None):
        # Sum over all indices
        super(PowerNode, self).__init__(parent, name=name)
        self.power = power

    def function(self, in_value):
        out_value = in_value ** self.power
        return out_value

    def derivative_function(self, in_value):
        out_value = self.power * (in_value ** (self.power - 1))
        return out_value


class ScalarMultiplyNode(FunctionNode):
    def __init__(self, parent, scalar, name=None):
        # Sum over all indices
        super(ScalarMultiplyNode, self).__init__(parent, name=name)
        self.scalar = self.type(scalar)

    def function(self, in_value):
        out_value = in_value * self.scalar
        return out_value

    def derivative_function(self, in_value):
        out_value = np.ones_like(in_value, dtype=self.type) * self.scalar
        return out_value

class ScalarAddNode(FunctionNode):
    def __init__(self, parent, scalar, name=None):
        # Sum over all indices
        super(ScalarAddNode, self).__init__(parent, name=name)
        self.scalar = self.type(scalar)

    def function(self, in_value):
        out_value = in_value + self.scalar
        return out_value

    def derivative_function(self, in_value):
        out_value = np.ones_like(in_value, dtype=self.type)
        return out_value


# Functions which return nodes for select operations which cannot be implemented by operators

def broadcast(in_node, out_shape):
    return BroadcastNode(in_node, out_shape)


def sigmoid(in_node):
    return SigmoidNode(in_node)


def reLu(in_node, leak_constant=0):
    return ReLuNode(in_node, leak_constant=leak_constant)


def reshape(in_node, shape):
    return ReshapeNode(in_node, shape)


def transpose(in_node, indices):
    return TransposeNode(in_node, indices)


def dot(parent1, parent2, indices_1, indices_2):
    return TensorDotNode(parent1, parent2, indices_1, indices_2)


def sum(parent, indices=None):
    return TensorSumNode(parent, indices=indices)