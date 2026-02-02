#!/usr/bin/env python3
"""Creates a neural network layer using TensorFlow."""
import tensorflow as tf


def create_layer(prev, n, activation):
    """Create a fully connected layer with variance scaling initialization."""
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.dense(
        prev,
        n,
        activation=activation,
        kernel_initializer=initializer,
        name="layer"
    )
    return layer