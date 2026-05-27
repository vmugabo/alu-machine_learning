#!/usr/bin/env python3
"""Neural Style Transfer - Task 0: Initialize"""
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()


class NST:
    """Performs tasks for neural style transfer."""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize NST instance."""
        if (not isinstance(style_image, np.ndarray)
                or style_image.ndim != 3
                or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if (not isinstance(content_image, np.ndarray)
                or content_image.ndim != 3
                or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """Rescales image so pixels are in [0, 1] and max side is 512 px."""
        if (not isinstance(image, np.ndarray)
                or image.ndim != 3
                or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")
        h, w = image.shape[0], image.shape[1]
        if h > w:
            new_h, new_w = 512, int(w * 512 / h)
        else:
            new_h, new_w = int(h * 512 / w), 512
        image = tf.constant(image, dtype=tf.float32)
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bicubic(image, [new_h, new_w])
        image = image / 255.0
        return tf.clip_by_value(image, 0.0, 1.0)