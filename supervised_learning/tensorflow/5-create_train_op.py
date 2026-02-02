#!/usr/bin/env python3
"""Creates the training operation for the network."""
import tensorflow as tf


def create_train_op(loss, alpha):
    """Create the training operation for the network using gradient descent"""
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train_op = optimizer.minimize(loss)
    return train_op
