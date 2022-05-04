import numpy as np
from scipy.optimize import minimize as scipy_minimize
from time import time

from ottodiff.exceptions import BroadcastingException
from ottodiff.primitives import ComputationGraph, tensor, reset_default_graph, DefaultGraph
from ottodiff.operations import ComputeNode, BroadcastNode, SumNode, TensorDotNode, sigmoid, sum
from ottodiff.optimization import minimize_gradient_descent
from ottodiff.nn import ReLuNode


def tensor_test():
    print("*****Tensor Test*****")
    t1_val = np.arange(2 * 3 * 4 * 5).reshape(2,3, 4, 5)
    t2_val = np.arange(4 * 5 * 6 * 7).reshape(4, 5, 6, 7)

    t1 = tensor((2, 3), (4, 5), value=t1_val)
    t2 = tensor((4, 5), (6, 7), value=t2_val)

    t3 = t1 * t2
    t4 = 10 * t1
    t5 = t1 * 10
    t6 = t1.transpose()
    print(t3.shape)
    print(t4.shape)
    print(t5.shape)
    print(t6.shape)


def test_broadcasting():
    print("*****Broadcasting Test*****")
    graph = ComputationGraph()
    n1 = ComputeNode(graph=graph, name='n1', value=np.ones((3, 1, 4)))
    try:
        b1 = BroadcastNode(n1, (4, 4, 4))
    except BroadcastingException:
        pass
    b1 = BroadcastNode(n1, (8, 3, 4, 4))
    graph.forward()
    print(b1.value().shape)
    graph.backward(b1)
    print(b1.derivative(n1).shape)

    graph = ComputationGraph()
    n1 = ComputeNode(graph=graph, name='n1', value=np.ones(1))
    b1 = BroadcastNode(n1, (2, 3))
    graph.forward()
    graph.backward(b1)
    print(b1.value())
    print(b1.derivative(n1).value)

    graph = ComputationGraph()
    n1 = ComputeNode(graph=graph, name='n1', value=np.array([[1, 2, 3]]))
    b1 = BroadcastNode(n1, (2, 1, 3))
    graph.forward()
    graph.backward(b1)
    print(b1.value())
    print(b1.derivative(n1).array())


# a1 = n1 + n2
# 3 = 1 + 3
def test_add():
    print("*****Add Test*****")
    graph = ComputationGraph()
    n1 = ComputeNode(graph=graph, name='n1', value=1)
    n2 = ComputeNode(graph=graph, name='n2', value=2)
    a1 = n1 + n2
    graph.forward()
    graph.check_all()
    print(a1.value())
    assert np.all(a1.value() == 3)


def test_sum():
    print("*****Sum Test*****")
    graph = ComputationGraph()
    n1 = ComputeNode(graph=graph, name='n1', value=1)
    n2 = ComputeNode(graph=graph, name='n2', value=np.array([[1, 2, 3], [4, 5, 6]]))
    n3 = ComputeNode(graph=graph, name='n3', value=np.array([[[100]], [[200]]]))

    a1 = SumNode([n1, n2, n3])
    a2 = a1 + n1
    graph.forward()
    graph.check_all()
    graph.backward(a1)
    graph.backward(a2)
    print(a1.value())
    print(a2.value())
    print(a1.derivative(n1).array())
    print(a2.derivative(n1).array())
    print(a2.derivative(n2).array())
    print(a2.derivative(n3).array())


def test_mult():
    print("*****Mult Test*****")
    graph = ComputationGraph()
    n1 = ComputeNode(graph=graph, name='n1', value=10)
    n2 = ComputeNode(graph=graph, name='n2', value=np.array([[1, 2, 3], [4, 5, 6]]))
    n3 = ComputeNode(graph=graph, name='n3', value=np.array([[[11]], [[31]]]))
    a1 = n1 * n2 * n3
    a2 = n2 * n2
    graph.forward()
    graph.check_all()
    graph.backward(a1)
    graph.backward(a2)
    print("Values")
    print(a1.value())
    print(a2.value())
    print("Derivatives")
    print(a1.derivative(n1).array())
    print(a1.derivative(n2).array())
    print(a1.derivative(n3).array())
    print(n2.derivatives)


# a1 = n1 * (n2 + n3 * (n4 + n5))
# 290 = 10 * (2 + 3 * (4 + 5))
def test_add_mult():
    print("*****Add/Mult Test*****")
    graph = ComputationGraph()
    n1 = ComputeNode(graph=graph, name='in_1', value=10)
    n2 = ComputeNode(graph=graph, name='in_2', value=2 * np.ones((2, 2)))
    n3 = ComputeNode(graph=graph, name='in_3', value=3 * np.ones((2, 1)))
    n4 = ComputeNode(graph=graph, name='in_4', value=4 * np.ones((3, 1, 1)))
    n5 = ComputeNode(graph=graph, name='in_5', value=5 * np.ones((1, 1, 2)))
    a1 = n1 * (n2 + n3 * (n4 + n5))
    graph.forward()
    graph.backward(a1)
    graph.check_all()
    print(a1.value())
    print("Derivatives")
    print(a1.derivative(n1).array())
    print(a1.derivative(n2).array())
    print(a1.derivative(n3).array())
    print(a1.derivative(n4).array())
    print(a1.derivative(n5).array())


def test_tensor_dot():
    print("*****Tensordot Test*****")
    # Vector matrix product
    graph = ComputationGraph()
    n1 = ComputeNode(graph=graph, name='in_1', value=np.arange(100).reshape(10, 10))
    n2 = ComputeNode(graph=graph, name='in_2', value=np.arange(10))

    a1 = TensorDotNode(n1, n2, [0], [0])
    graph.forward()
    graph.backward(a1)
    print(a1.value().shape)

    # Inner produce
    graph = ComputationGraph()
    n1 = ComputeNode(graph=graph, name='in_1', value=np.arange(100).reshape(10, 10) + 1)
    n2 = ComputeNode(graph=graph, name='in_2', value=np.arange(100).reshape(10, 10))

    a1 = TensorDotNode(n1, n2, [0, 1], [0, 1])
    graph.forward()
    graph.backward(a1)
    print(a1.value().shape)

    #Crazy tensor product
    graph = ComputationGraph()
    n1 = ComputeNode(graph=graph, name='in_1', value=np.arange(3 * 5 * 7 * 2 * 2 * 4).reshape(3, 5, 7, 2, 2, 4))
    n2 = ComputeNode(graph=graph, name='in_2', value=np.arange(7 * 1 * 2 * 3 * 5).reshape(7, 1, 2, 3, 5))

    a1 = TensorDotNode(n1, n2, [0, 2, 4], [3, 0, 2])
    graph.forward()
    graph.backward(a1)
    assert a1.value().shape == (5, 2, 4, 1, 5)
    for inidx1 in range(5):
        for outidx1 in range(5):
            for inidx2 in range(2):
                for outidx2 in range(2):
                    for inidx3 in range(4):
                        for outidx3 in range(4):
                            d_slice = a1.derivative(n1).array()[outidx1, outidx2, outidx3, :, :, :, inidx1, :, inidx2, :, inidx3]
                            if (inidx1, inidx2, inidx3) == (outidx1, outidx2, outidx3):
                                d_slice = d_slice.transpose([3, 0, 4, 2, 1])
                                assert np.all(d_slice == n2.value())
                            else:
                                assert np.all(d_slice == 0)
    for inidx1 in range(1):
        for outidx1 in range(1):
            for inidx2 in range(5):
                for outidx2 in range(5):
                    d_slice = a1.derivative(n2).array()[:, :, :, outidx1, outidx2, :, inidx1, :, :, inidx2]
                    if (inidx1, inidx2) == (outidx1, outidx2):
                        d_slice = d_slice.transpose([5, 0, 3, 1, 4, 2])
                        assert np.all(d_slice == n1.value())
                    else:
                        assert np.all(d_slice == 0)
    # Matrix-matrix product
    graph = ComputationGraph()
    n1 = ComputeNode(graph=graph, name='in_1', value=np.arange(3 * 5 ).reshape(3, 5))
    n2 = ComputeNode(graph=graph, name='in_2', value=np.arange(5 * 4).reshape(5, 4))

    a1 = TensorDotNode(n1, n2, [1], [0])
    graph.forward()
    graph.backward(a1)
    print(n1.shape)
    print(n2.shape)
    print(a1.shape)
    print(a1.derivative(n1).array().shape)


def test_transpose():
    print("*****Transpose Test*****")
    graph = ComputationGraph()
    n1 = ComputeNode(graph=graph, name='in_1', value=np.arange(3 * 5 * 7).reshape(3, 5, 7))

    a1 = n1.transpose((1, 0, 2))
    graph.forward()
    graph.backward(a1)

    assert np.all(a1.value() == n1.value().transpose((1, 0, 2)))
    derivative = a1.derivative(n1).array()
    for index, value in np.ndenumerate(derivative):
        if index[0] == index[4] and index[1] == index[3] and index[2] == index[5]:
            assert value == 1
        else:
            assert not value


def test_reshape():
    print("*****Reshape Test*****")
    graph = ComputationGraph()
    n1 = ComputeNode(graph=graph, name='in_1', value=np.arange(3 * 4 * 1).reshape(3, 4, 1))

    a1 = n1.reshape((6, 1, 2))
    graph.forward()
    graph.backward(a1)

    assert np.all(a1.value() == n1.value().reshape((6, 1, 2)))
    derivative = a1.derivative(n1).array()
    print(derivative.shape)
    for index, value in np.ndenumerate(derivative):
        if 2 * index[0] + 2 * index[1] + index[2] == 4 * index[3] + index[4] + index[5]:
            assert value == 1
        else:
            assert not value


def test_tensor_sum():
    print("*****Tensor Sum Test*****")
    graph = ComputationGraph()
    n1 = ComputeNode(graph=graph, name='in_1', value=np.arange(3 * 5 * 7 * 2).reshape(3, 5, 7, 2))

    a1 = n1.sum()
    a2 = n1.sum([0, 1])
    graph.forward()
    graph.backward(a1)
    graph.backward(a2)
    assert a1.derivative(n1).shape == (1, 3, 5, 7, 2)
    assert a2.derivative(n1).shape == (7, 2, 3, 5, 7, 2)
    assert np.all(a1.derivative(n1).array() == 1)

    for index, value in np.ndenumerate(a2.derivative(n1).array()):
        if index[0] == index[4] and index[1] == index[5]:
            assert value == 1
        else:
            assert not value


def test_relu():
    print("*****ReLu Test*****")
    graph = ComputationGraph()
    n1 = ComputeNode(graph=graph, name='in_1', value=np.arange(4 * 5 * 7).reshape(4, 5, 7) - 70)

    a1 = ReLuNode(n1)
    a2 = ReLuNode(n1, leak_constant=.01)

    graph.forward()
    graph.backward([a1, a2])
    assert np.all(a1.value()[n1.value() >= 0] == n1.value()[n1.value() >= 0])
    assert np.all(a1.value()[n1.value() < 0] == 0)
    assert np.all(a2.value()[n1.value() >= 0] == n1.value()[n1.value() >= 0])
    assert np.all(a2.value()[n1.value() < 0] == n1.value()[n1.value() < 0] * .01)


def test_minimize():
    print("*****Minimize Test*****")
    graph = ComputationGraph()
    n1 = ComputeNode(graph=graph, name='n1', value=np.arange(10))
    n2 = ComputeNode(graph=graph, name='n2', value=0)

    lossnode = n1 - n2

    minimize_gradient_descent(lossnode, [n2], step=.001, max_iter=1000)
    assert np.all(np.abs(n2.value() - 4.5) < 1e-4)

    # Basic linear regression
    graph = ComputationGraph()
    data = ComputeNode(graph=graph, name='n1', value=np.random.normal(0, 1, (100, 10)))
    x = ComputeNode(graph=graph, name='n2', value=np.random.normal(0, 1, (10,)))
    y = ComputeNode(graph=graph, name='n2', value=np.random.normal(0, 1, (100,)))

    lossnode = y - data.dot(x, [1], [0])

    graph.forward()
    print("Initial loss %f" % np.sum(lossnode.value()**2))
    minimize_gradient_descent(lossnode, [x], step=.001, max_iter=100)
    print("Final loss %f" % np.sum(lossnode.value()**2))
    print("OttoDiff solution: ")
    print(x.value())

    def func(x):
        return np.linalg.norm(y.val - np.dot(data.val, x))
    res = scipy_minimize(func, np.zeros(10))
    print("Scipy solution: ")
    print(res['x'])
    assert np.all(np.abs(x.value() - res['x']) < 1e-4)


def test_gradient_speed():
    print("*****Speed Test*****")
    reset_default_graph()
    t1 = time()
    n1 = ComputeNode(name='in_1', value=np.arange(3 * 50 * 7 * 2 * 2 * 4).reshape(3, 50, 7, 2, 2, 4) + 1)
    n2 = ComputeNode(name='in_2', value=np.arange(7 * 1 * 2 * 3 * 50).reshape(7, 1, 2, 3, 50) + 1)
    n3 = ComputeNode(name='in_3', value=np.arange(50 * 2 * 4 * 1 * 50).reshape(50, 2, 4, 1, 50) + 1)

    a1 = TensorDotNode(n1, n2 ** 2, [0, 2, 4], [3, 0, 2]) # shape 5, 2, 4, 1, 5

    a2 = sigmoid(n3 ** -2)

    a3 = a1 + a2
    a4 = sum(a3, [0, 1])
    t2 = time()
    DefaultGraph.forward()
    t3 = time()
    DefaultGraph.backward(a4)
    t4 = time()
    print("Build: ", t2 - t1)
    print("Evaluate ", t3 - t2)
    print("Differentiate", t4 - t3)
    print(a4.value())
    DefaultGraph.print_graph()


tensor_test()

test_add()
test_sum()
test_mult()
test_add_mult()
test_tensor_dot()
test_broadcasting()
test_tensor_sum()
test_reshape()
test_transpose()

test_relu()

test_minimize()

test_gradient_speed()
