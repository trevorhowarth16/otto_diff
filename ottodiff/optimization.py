import numpy as np


def minimize_gradient_descent(loss_node, input_nodes, step=.001, max_iter=10000, verbose=False):
    if loss_node.shape != (1,):
        loss = (loss_node ** 2).sum()
    else:
        loss = loss_node ** 2

    graph = loss.graph
    for i in range(max_iter):
        graph.forward()
        graph.backward(loss)
        if verbose:
            print("Iteration %d: Loss = %0.3f" % (i, loss.value()[0]))
        for node in input_nodes:
            derivative = loss.derivative(node)
            d_value = derivative.array().reshape(derivative.lower_shape)
            node.set_value(node.value() - step * d_value)