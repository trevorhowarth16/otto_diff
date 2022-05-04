# Basic NN training
import numpy as np
from ottodiff.nn import CrossEntropyLossNode, InputLayer, LinearLayer, normalize
from ottodiff.primitives import DefaultGraph
from ottodiff.optimization import minimize_gradient_descent
from ottodiff.operations import SigmoidNode

# Load dataset
# This is the Pima Indian diabetes dataset from: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
all_data = np.loadtxt('diabetes_dataset.csv', delimiter=',')
x_train = all_data[:200, :-1]
y_train = all_data[:200, -1].reshape(-1, 1)
x_test = all_data[200:400, :-1]
y_test = all_data[200:400, -1].reshape(-1, 1)

normed_x_train, (m, s) = normalize(x_train)
normed_x_test, _ = normalize(x_test, m, s)
num_examples, in_dim = x_train.shape

# Define network
in_layer = InputLayer(x_train.shape)
layer_1 = LinearLayer(in_dim, 10)
layer_2 = LinearLayer(10, 20)
sig_layer = LinearLayer(20, 1, activation=SigmoidNode)
labels_layer = InputLayer(y_train.shape)


# Build graph
activations0 = layer_1(in_layer.output)
activations1 = layer_2(activations0)
logits = sig_layer(activations1 * 0.01)
loss = CrossEntropyLossNode(logits, labels_layer.output)

# Compute initial accuracy
in_layer.feed(normed_x_train)
labels_layer.feed(y_train)
DefaultGraph.forward()
print('Untrained accuracy on training data = %03f' % np.mean((logits.value() > 0.5) == (labels_layer.output.value() > 0.5)))
in_layer.feed(normed_x_test)
labels_layer.feed(y_test)
DefaultGraph.forward()
print('Untrained accuracy on test data = %03f' % np.mean((logits.value() > 0.5) == (labels_layer.output.value() > 0.5)))
print('\n')

# Train
minimize_gradient_descent(loss, DefaultGraph.get_mutable_nodes(), max_iter=100, verbose=True)

# Compute final accuracy
print('\n')
in_layer.feed(normed_x_train)
labels_layer.feed(y_train)
DefaultGraph.forward()
print('Trained accuracy on training data = %03f' % np.mean((logits.value() > 0.5) == (labels_layer.output.value() > 0.5)))
in_layer.feed(normed_x_test)
labels_layer.feed(y_test)
DefaultGraph.forward()
print('Trained accuracy on test data = %03f' % np.mean((logits.value() > 0.5) == (labels_layer.output.value() > 0.5)))
