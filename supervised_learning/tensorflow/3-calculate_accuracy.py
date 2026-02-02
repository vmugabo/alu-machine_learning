#!/usr/bin/env python3
"""Calculates the accuracy of a prediction."""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Calculate the accuracy of a prediction"""
    correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy
