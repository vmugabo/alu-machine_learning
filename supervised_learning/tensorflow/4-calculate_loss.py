#!/usr/bin/env python3
"""Calculates the softmax cross-entropy loss of a prediction."""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """Calculate the softmax cross-entropy loss of a prediction"""
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss
