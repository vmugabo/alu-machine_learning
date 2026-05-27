#!/usr/bin/env python3
"""Neural Style Transfer - Task 3: Extract Features"""
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
        self.load_model()
        self.generate_features()

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

    def load_model(self):
        """Creates the Keras model used to calculate cost."""
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet')
        vgg.trainable = False

        x = vgg.input
        model_outputs = []
        target_layers = self.style_layers + [self.content_layer]

        for layer in vgg.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                x = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name
                )(x)
            else:
                layer.trainable = False
                x = layer(x)

            if layer.name in target_layers:
                model_outputs.append(x)

            if layer.name == self.content_layer:
                break

        self.model = tf.keras.Model(inputs=vgg.input, outputs=model_outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """Calculates the gram matrix of a layer output."""
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable))
                or len(input_layer.shape) != 4):
            raise TypeError("input_layer must be a tensor of rank 4")

        shape = tf.shape(input_layer)
        h, w = shape[1], shape[2]
        c = input_layer.shape[3]

        F = tf.reshape(input_layer, (h * w, c))
        gram = tf.matmul(F, F, transpose_a=True)
        gram = gram / tf.cast(h * w, tf.float32)
        return tf.expand_dims(gram, 0)

    def generate_features(self):
        """Extracts style and content features from the model."""
        vgg19 = tf.keras.applications.vgg19

        style_prep = vgg19.preprocess_input(self.style_image * 255)
        style_outputs = self.model(style_prep)
        self.gram_style_features = [self.gram_matrix(out)
                                    for out in style_outputs[:-1]]

        content_prep = vgg19.preprocess_input(self.content_image * 255)
        content_outputs = self.model(content_prep)
        self.content_feature = content_outputs[-1]