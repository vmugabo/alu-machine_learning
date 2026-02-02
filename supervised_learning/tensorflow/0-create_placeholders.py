#!/usr/bin/env python3
"""Creates placeholders for input and output tensors in TensorFlow."""
import tensorflow as tf


def create_placeholders(nx, classes):
    """Create placeholders for input features and target classes."""
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")

    return x, y
