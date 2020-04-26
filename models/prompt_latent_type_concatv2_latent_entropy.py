import tensorflow as tf
import numpy as np
import sys


class PromptLatentTypeConcatModelV2LatentEntropy(object):
	"""
	A CNN for text classification.
	Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.

	To keep things modular, both question & response sequences are parameters, but only response
	features are used.
	"""
	def __init__(
	  self, conversation_length, num_channels, embedding_size, num_hidden_layers, regularization_coefficient):

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
			self.batch_size = tf.shape(self.input_prompts)[0]
			attention_dims = 100

			single_batch = tf.reshape(self.input_prompts, shape=[self.batch_size * conversation_length, int(embedding_size)])


			self.W_q = tf.Variable(tf.truncated_normal([embedding_size, num_channels]), name="W_q")
			self.b_q = tf.Variable(tf.constant(0.1, shape=[num_channels]), name="b_q")
			

			query = tf.nn.xw_plus_b(tf.nn.dropout(single_batch, keep_prob=self.dropout_keep
				), self.W_q, self.b_q, name="query_layer")
			query_3D = tf.reshape(query, [self.batch_size, conversation_length, num_channels])
			self.usefulness = tf.math.softmax(query_3D, axis=2, name="channel_saliences")
						
			self.usefulness = tf.clip_by_value(self.usefulness, tf.constant(1e-15), tf.constant(1.0))
			
			usefulness_dropout = tf.nn.dropout(self.usefulness, keep_prob=1)
			

			channel_evidence = []

			prompt_evidence = 1.0 / tf.reduce_sum(tf.cast(self.input_masks[:, :, tf.newaxis], dtype=tf.float32)) * tf.reduce_sum(self.input_prompts, axis=1)
			channel_evidence.append(prompt_evidence)

			for channel in range(num_channels):
				norm = 1.0 / (tf.reduce_sum(usefulness_dropout[:, :, channel, tf.newaxis] * tf.cast(self.input_masks[:, :, tf.newaxis], dtype=tf.float32), axis=1))
				r_evidence = norm * tf.reduce_sum(
					# don't technically need to apply masks here because input responses are 0 when beyond M_i
					usefulness_dropout[:, :, channel, tf.newaxis] * self.input_responses,
					axis=1
				)
				
				channel_evidence.append(r_evidence)

			

			combined_evidence = tf.concat(axis=1, values=channel_evidence)
			combined_evidence = tf.reshape(combined_evidence, [self.batch_size, embedding_size * (1 + num_channels)])


			if num_hidden_layers > 0:
				combined_evidence = tf.layers.dense(
				inputs = tf.nn.dropout(combined_evidence, keep_prob=self.dropout_keep),
				units = embedding_size,
				name="scores_hidden_layer",
				activation = tf.nn.relu,
				kernel_initializer=tf.contrib.layers.xavier_initializer(seed=43),
			)   

			self.scores = tf.layers.dense(
				inputs = tf.nn.dropout(combined_evidence, keep_prob=self.dropout_keep),
				units = 2,
				name="scores_layer",
				kernel_initializer=tf.contrib.layers.xavier_initializer(seed=45),
			)   

								   

			self.predictions = tf.argmax(self.scores, 1, name="compute_predictions")
			   
		# Calculate mean cross-entropy loss
		with tf.name_scope("loss"):
			losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)

			#adjusted_usefulness = usefulness_dropout[:, :, :] * tf.cast(self.input_masks[:, :, tf.newaxis], dtype=tf.float32)

			norm = 1.0 / tf.reduce_sum(tf.reduce_sum(tf.cast(self.input_masks[:, :], dtype=tf.float32), axis=1), axis=0)
			
			entropy_component = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(usefulness_dropout * tf.math.log(usefulness_dropout), axis=2) * 
				tf.cast(self.input_masks[:, :], dtype=tf.float32), axis=1), axis=0)      
			

			self.loss = tf.reduce_mean(losses) - regularization_coefficient * norm * entropy_component
			#self.loss = tf.reduce_mean(losses)

		# Accuracy
		with tf.name_scope("accuracy"):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")





