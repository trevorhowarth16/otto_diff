import numpy as np
from ottodiff.exceptions import BroadcastingException


# Utilities for processing tensor shapes
def broadcast_shapes(shapes):
    output_shape = []
    max_dimension = np.max([len(x) for x in shapes])
    # Loop backwards through indices
    for idx in range(-max_dimension, 0)[::-1]:
        max_dim = 1
        for shape in shapes:
            if len(shape) >= -idx:
                dim = shape[idx]
                if max_dim == 1:
                    max_dim = dim
                else:
                    if dim != 1 and dim != max_dim:
                        print('Cannot broadcast %d to %d on %dth dimension' % (dim, max_dim, max_dimension + idx))
                        raise BroadcastingException
        output_shape.append(max_dim)

    return tuple(output_shape[::-1])


def flatten_shape(shape):
    total_dim = 1
    for dim in shape:
        total_dim *= dim
    return total_dim