import tensorflow as tf
import numpy as np
import sys

class ResponseBaseline(object):
	"""
	A CNN for text classification.
	Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.

	To keep things modular, both question & response sequences are parameters, but only response
	features are used.
	"""
	def __init__(
	  self, conversation_length,
	  embedding_size):

		num_classes = 2

		# Placeholders for input, output and dropout
		self.input_prompts = tf.placeholder(tf.float32, [None, conversation_length, embedding_size], name="input_prompts")
		self.input_responses = tf.placeholder(tf.float32, [None, conversation_length, embedding_size], name="input_responses")
		self.input_masks = tf.placeholder(tf.float32, [None, conversation_length])
		self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
		self.dropout_keep = tf.placeholder(tf.float32, name="dropout_keep")

		# Keeping track of l2 regularization loss (optional)
		l2_loss = tf.constant(0.0)


		
		with tf.variable_scope("classifier"):
			with tf.variable_scope("output"):
				self.batch_size = tf.shape(self.input_prompts)[0]
			
				self.normalizer = (1.0 / tf.cast(tf.reduce_sum(self.input_masks, axis=1),dtype=tf.float32))[:, tf.newaxis] 
				
				print('\n================Using baseline================\n')
				self.evidence = self.normalizer * tf.reduce_sum(self.input_responses, axis=1)



				self.logits = tf.nn.dropout(tf.layers.dense(
					inputs = self.evidence,					
					kernel_initializer=tf.contrib.layers.xavier_initializer(seed=42),
					units = 2,
					name="logits_layer"), keep_prob=1)


				self.predictions = tf.argmax(self.logits, 1, name="predictions")

			# Calculate mean cross-entropy loss
			with tf.name_scope("loss"):
				losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
				self.loss = tf.reduce_mean(losses)

			# Accuracy
			with tf.name_scope("accuracy"):
				correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
				self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
