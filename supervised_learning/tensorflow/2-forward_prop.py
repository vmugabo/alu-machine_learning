#!/usr/bin/env python3
"""Builds forward propagation graph for neural network."""
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Create forward propagation graph for the neural network"""
    output = x
    for i in range(len(layer_sizes)):
        output = create_layer(output, layer_sizes[i], activations[i])
    return output
