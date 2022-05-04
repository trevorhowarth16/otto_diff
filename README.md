# otto_diff

Ottodiff if a fully python graph-based autodifferentiation framework. Mathematical operations are represented as nodes and arranged into an overarching computation graph. Each node type defines its own derivative and derivatives of composite operations in the graph are calculated using the chain rule. When the package is loaded a computation graph is automatically instantiated and can be accessed as ottodiff.DefaultGraph.

Values can be fed in at the top of the graph and forward propagated to perform calculations. After forward propagation, backpropagation can be run and the derivative of values in the graph can be taken relative to any output. Among other things, this can be used for building neural networks and a basic set of utilities have been developed for building classifiers (see nn_example.py). The values of all nodes are represented using sparse matrices and scipy.sparse is used for many of the actual calculations. Derivatives are represented as tensors that have upper and lower indices (similar to tensors use by physicists). If we are calculating dY/dX, the derivative's upper indices represent the shape of Y and the derivative's lower indices represent the shape of X. Further examples of how to use the framework can be found in tests.py.

# Development TODOs
- Allow node shapes to change dynamically (enables variable batch sizes for nns)
- Batchnorm layer
- Improve handling of scalars
- Graph/nn serialization (and load with different batch size)
- Second derivatives
- Implement more optimization algorithms (Rmsprop, Adam, Levenberg–Marquardt)
