import features_provider
import data_provider
from models.response_baseline import ResponseBaseline
from models.prompt_baseline import PromptBaseline
from models.prompt_latent_type import PromptLatentTypeModel
from models.prompt_latent_type_concat import PromptLatentTypeConcatModel
from models.promptresponse_baseline import PromptResponseBaseline
from models.prompt_latent_type_concatv2 import PromptLatentTypeConcatModelV2
from models.prompt_latent_type_latent_entropy import PromptLatentTypeModelLatentEntropy
from models.prompt_latent_type_concat_latent_entropy import PromptLatentTypeConcatModelLatentEntropy
from models.prompt_latent_type_concatv2_latent_entropy import PromptLatentTypeConcatModelV2LatentEntropy
from sklearn.metrics import f1_score
import random
import numpy as np
import tensorflow as tf
import os
import time
import datetime
import math
from enum import Enum
from exclude_question_set import exclude_questions_from_set

class SetTypes(Enum):
	Train = 1
	Dev = 2
	Test = 3



np.random.seed(7)

tf.flags.DEFINE_boolean('write_summaries', False, 'Whether to write gradient summaries, takes up much space')
tf.flags.DEFINE_boolean('save_final_epoch', False, 'Whether to use the maximum or final dev score for time vessel')


tf.flags.DEFINE_string('model', '', 'Describer for which of the models to test')
tf.flags.DEFINE_string('embedding_source', 'glove', 'Which pretrained embeddings to use')
tf.flags.DEFINE_integer("num_tests", 10, "Number of training epochs (default: 200)")
tf.flags.DEFINE_string('experiment_prefix', '', 'Subfolder in which to save checkpoints')
tf.flags.DEFINE_float('train_size', 0.6, 'Percent of data to use for training')
tf.flags.DEFINE_float('dev_size', 0.25, 'Percent of remaining data to use for development scoring')
tf.flags.DEFINE_integer("batch_size", 10, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 3, "Number of epochs between early stop evaluations")

tf.flags.DEFINE_integer("prompt_latent_type__num_channels", 2, "Number of latent types in the model")
tf.flags.DEFINE_float("prompt_latent_type__dropout_keep", 1, "Probability of not dropping out")

tf.flags.DEFINE_integer("promptresponse_complex__num_hidden_layers", 0, "Number of hidden layers in the complex baseline")

tf.flags.DEFINE_float('entropy_coefficient', 0.1, 'Test')
tf.flags.DEFINE_float('learning_rate', 1e-3, 'Test')

tf.flags.DEFINE_integer("conversation_length", 100, "Max number of turns in a conversation")
tf.flags.DEFINE_integer("embedding_size", 100, "Number of training epochs (default: 200)")




FLAGS = tf.flags.FLAGS



def get_model(model_id):
	if model_id == "response_baseline":
		attn = ResponseBaseline(
			conversation_length=100,			
			embedding_size=FLAGS.embedding_size								
		)
	elif model_id == "prompt_baseline":
		attn = PromptBaseline(
			conversation_length=100,
			embedding_size=FLAGS.embedding_size
		)	
	elif model_id == "promptresponse_baseline":
		attn = PromptResponseBaseline(
			conversation_length=100,
			embedding_size=FLAGS.embedding_size
		)	
	elif model_id == "prompt_latent_type_latent_entropy":
		attn = PromptLatentTypeModelLatentEntropy(
			conversation_length=100,
			embedding_size=FLAGS.embedding_size,
			num_channels=FLAGS.prompt_latent_type__num_channels,
			regularization_coefficient=FLAGS.entropy_coefficient
		)
	elif model_id == "prompt_latent_type_concat_latent_entropy":
		attn = PromptLatentTypeConcatModelLatentEntropy(
			conversation_length=100,
			embedding_size=FLAGS.embedding_size,
			num_channels=FLAGS.prompt_latent_type__num_channels,
			num_hidden_layers=FLAGS.promptresponse_complex__num_hidden_layers,
			regularization_coefficient=FLAGS.entropy_coefficient
		)
	elif model_id == "prompt_latent_type_concatv2_latent_entropy":
		attn = PromptLatentTypeConcatModelV2LatentEntropy(
			conversation_length=100,
			embedding_size=FLAGS.embedding_size,
			num_channels=FLAGS.prompt_latent_type__num_channels,
			num_hidden_layers=FLAGS.promptresponse_complex__num_hidden_layers,
			regularization_coefficient=FLAGS.entropy_coefficient
		)
	else:
		print("\nNO SUCH MODEL\n")
		exit()
	return attn


def train():
	timestamp = str(int(time.time()))
	examples, labels, refs = data_provider.generate_data(verbose=True, return_indices=True)	

	labels = np.array(labels)	
	examples = exclude_questions_from_set(examples, "remove_questions_binary.txt")
        #it is not an issue that this doesn't use the bert question file since the embeddings were already generated

	embed_source = FLAGS.embedding_source
	if (embed_source == "glove"):
		prompt_features, response_features, masks = features_provider.get_features(examples, conversation_length=FLAGS.conversation_length, embedding_size=FLAGS.embedding_size)
	elif (embed_source == "bert"):
		prompt_features, response_features, masks = features_provider.get_bert_features()

	# Output directory for models and summaries
	
	out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))
	out_dir = os.path.abspath(os.path.join(out_dir, "{}".format(FLAGS.experiment_prefix), timestamp))
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	print("Writing to {}\n".format(out_dir))


	#Get random splits
	random.seed(7)
	split_seeds = [random.randint(1, 100000) for _ in range(FLAGS.num_tests)]

	splits_file = open('splits_file.csv', 'w')
	for test_num in range(FLAGS.num_tests):


		time_vessel_out_dir = os.path.join(out_dir, 'time_vessels', str(test_num))
		if not os.path.exists(time_vessel_out_dir):
			os.makedirs(time_vessel_out_dir)


		train_indices, dev_indices, test_indices = data_provider.shuffle_stratified_split([i for i in range(len(examples))], labels, 
			train_size=FLAGS.train_size, dev_size=FLAGS.dev_size, seed=split_seeds[test_num])

		splits_file.write('Split {}\n'.format(test_num))
		splits_file.write('Train: {}\n'.format(','.join([str(i) for i in train_indices])))
		splits_file.write('Dev: {}\n'.format(','.join([str(i) for i in dev_indices])))
		splits_file.write('Test: {}\n\n'.format(','.join([str(i) for i in test_indices])))


		if test_num == 9:
			splits_file.close()


		early_stop_f1 = float("-inf")	
		time_vessel = {}	

		with tf.Graph().as_default():             		
			tf.random.set_random_seed(777)		
			session_conf = tf.ConfigProto(
				allow_soft_placement=True,
				log_device_placement=False, inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
			sess = tf.Session(config=session_conf)

			with sess.as_default(): 
				print('Starting test {} training...'.format(test_num))

				model = get_model(FLAGS.model)

				# Define Training procedure
				global_step = tf.Variable(0, name="global_step", trainable=False)
				optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
				grads_and_vars = optimizer.compute_gradients(model.loss, var_list=tf.trainable_variables())

				train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)	


				#Set up training summary
				train_summary_dir = os.path.join(out_dir, "summaries", str(test_num), "train")
				train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
				train_summary_op = get_train_summary_op(train_summary_dir, model, grads_and_vars)

				
				# Initialize all variables
				sess.run(tf.global_variables_initializer())

				
				def get_predictions(set_type):
					indices = train_indices if set_type == SetTypes.Train else (
						dev_indices if set_type == SetTypes.Dev else test_indices)
					feed_dict = {
						model.input_prompts: prompt_features[indices],
						model.input_responses: response_features[indices],
						model.input_masks: masks[indices],
						model.input_y: data_provider.convert_to_one_hot(labels[indices]),
						model.dropout_keep: 1
					}
					return sess.run([model.predictions, model.accuracy], feed_dict)
						


				def get_early_stop_performance():										
					return f1_score(y_true=labels[dev_indices], y_pred=get_predictions(SetTypes.Dev)[0])				


				def capture_time_vessel(epoch):
					train_predictions, _ = get_predictions(SetTypes.Train)
					train_true = labels[train_indices]
					dev_predictions, _ = get_predictions(SetTypes.Dev)
					dev_true = labels[dev_indices]
					test_predictions, _ = get_predictions(SetTypes.Test)
					test_true = labels[test_indices]

					return {
						'epoch': epoch, 
						'train_predictions': train_predictions,
						'train_true': train_true,
						'dev_predictions': dev_predictions,
						'dev_true': dev_true,
						'test_predictions': test_predictions,
						'test_true': test_true
				}


				batches = data_provider.get_batches(train_indices, FLAGS.batch_size, FLAGS.num_epochs, shuffle=True)
				for batch_indices in batches:
					prompt_features_batch = prompt_features[batch_indices]
					response_features_batch = response_features[batch_indices]
					masks_batch = masks[batch_indices]
					labels_batch = labels[batch_indices]

					# train step
					feed_dict = {				
						model.input_prompts: prompt_features_batch,
						model.input_responses: response_features_batch,
						model.input_masks: masks_batch,
						model.input_y: data_provider.convert_to_one_hot(labels_batch),
						model.dropout_keep: FLAGS.prompt_latent_type__dropout_keep
					}

					_, step, summaries, loss, accuracy = sess.run(
								[train_op, global_step, train_summary_op, model.loss, model.accuracy],
								feed_dict)

					#Write summaries
					if FLAGS.write_summaries:
						train_summary_writer.add_summary(summaries, step)					

					
					current_step = tf.train.global_step(sess, global_step)

					#log training step
					time_str = datetime.datetime.now().isoformat()
					print("{}: step {}, loss {:g}, acc {:g}".format(test_num, step, loss, accuracy))


					# early stop time vessel
					batches_per_epoch = int((len(train_indices)-1)/FLAGS.batch_size) + 1
					if current_step % (FLAGS.evaluate_every * batches_per_epoch)  == 0:
						current_epoch = int(current_step / batches_per_epoch)
						current_performance = get_early_stop_performance()


						with open(os.path.join(time_vessel_out_dir, 'dev_f1_tracker.csv'), 'a') as dev_f1_tracker:
							dev_f1_tracker.write('{}\n'.format(current_performance))

						if current_performance > early_stop_f1:
							early_stop_f1 = current_performance		
							time_vessel = capture_time_vessel(current_epoch)

						_, set_accuracy = get_predictions(set_type=SetTypes.Train)
						train_summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="set accuracy", simple_value=set_accuracy)]), step)

				if FLAGS.save_final_epoch:
					time_vessel = capture_time_vessel(FLAGS.num_epochs)

					fnm = os.path.join(time_vessel_out_dir, 'saliences{}.csv'.format(test_num))
					open(fnm, 'w').close()




		save_time_vessel(time_vessel, test_num, time_vessel_out_dir)
	print('Wrote everything to {}'.format(out_dir))


def get_train_summary_op(train_summary_dir, model, grads_and_vars):
	grad_summaries = []
	for g, v in grads_and_vars:
		if g is not None:
			grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
			sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
			grad_summaries.append(grad_hist_summary)
			grad_summaries.append(sparsity_summary)
	grad_summaries_merged = tf.summary.merge(grad_summaries)

	loss_summary = tf.summary.scalar("batch loss", model.loss)
	acc_summary = tf.summary.scalar("batch accuracy", model.accuracy)
	train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])

	return train_summary_op


def get_train_set_summary_op(train_summary_dir, model, grads_and_vars):	
	acc_summary = tf.summary.scalar("set accuracy", model.accuracy)	
	return acc_summary


def save_time_vessel(time_vessel, test_num, out_dir):
	#out_dir = os.path.join(out_dir, 'time_vessels', str(test_num))

	#if not os.path.exists(out_dir):
	#	os.makedirs(out_dir)

	with open(os.path.join(out_dir, 'train_predictions.csv'), 'w') as dev_file:
		dev_file.writelines(['{},{}\n'.format(time_vessel['train_predictions'][i], time_vessel['train_true'][i]) for i in range(len(time_vessel['train_predictions']))])		
	with open(os.path.join(out_dir, 'dev_predictions.csv'), 'w') as dev_file:
		dev_file.writelines(['{},{}\n'.format(time_vessel['dev_predictions'][i], time_vessel['dev_true'][i]) for i in range(len(time_vessel['dev_predictions']))])
	with open(os.path.join(out_dir, 'test_predictions.csv'), 'w') as test_file:
		test_file.writelines(['{},{}\n'.format(time_vessel['test_predictions'][i], time_vessel['test_true'][i]) for i in range(len(time_vessel['test_predictions']))])
	with open(os.path.join(out_dir, 'vessel.info'), 'w') as vessel_file:
		vessel_file.write('epoch: {}\n'.format(time_vessel['epoch']))
		vessel_file.write('channels: {}\n'.format(FLAGS.prompt_latent_type__num_channels))
		vessel_file.write('dropout: {}\n'.format(FLAGS.prompt_latent_type__dropout_keep))
		vessel_file.write('entropy penalize: {}\n'.format(FLAGS.entropy_coefficient))

		quality_predictions = [pr for pr in time_vessel['test_predictions']]
		quality_predictions.extend([pr for pr in time_vessel['dev_predictions']])

		quality_true = [pr for pr in time_vessel['test_true']]
		quality_true.extend([pr for pr in time_vessel['dev_true']])

		quality = f1_score(y_true=quality_true, y_pred=quality_predictions)
		vessel_file.write('quality: {}\n'.format(quality))



def main(argv=None):     
	train()


if __name__ == '__main__':
	tf.app.run()





