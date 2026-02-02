#!/usr/bin/env python3
"""Evaluates the output of a neural network."""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """Evaluate the output of a neural network"""
    with tf.Session() as sess:
        # Import the meta graph
        saver = tf.train.import_meta_graph(save_path + '.meta')

        # Restore the session
        saver.restore(sess, save_path)

        # Get tensors from collection
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]

        # Evaluate the model
        prediction = sess.run(y_pred, feed_dict={x: X, y: Y})
        acc = sess.run(accuracy, feed_dict={x: X, y: Y})
        cost = sess.run(loss, feed_dict={x: X, y: Y})

        return prediction, acc, cost
